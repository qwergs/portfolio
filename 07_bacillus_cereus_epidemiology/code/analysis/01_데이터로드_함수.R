# ==============================================================================
# 01_데이터로드_함수.R — 질병·기간별 cor_merged 생성 (공통 변수 풀)
#   build_cor_merged(disease, ys, ye, include_food)
#   ★ 2011–2024 장기 대응:
#     - 파일별 단년 보간(apply_cf) 폐기 → 병합 후 '다년 선형보간' 1회 패스
#     - '스팬 게이트': 실측이 [ys+MAX_CARRY, ye-MAX_CARRY] 양끝을 못 덮는 변수는 메인 제외
#       (예: 1인가구 2015~, 손씻기 격년 일부, 식품 2019~ → 자동 드롭/로그)
#     - 내부 결측은 region별 선형보간, 양끝은 최대 MAX_CARRY년만 인접값 연장
# 사전: 00_config_공통.R 가 source 됨. df_raw 는 1회 캐시.
# ==============================================================================

if (!exists("DF_RAW_CACHE")) {
  cat("  ▶ 식중독최종.csv 로드 (1회)...\n")
  DF_RAW_CACHE <- read.csv(PATH_DISEASE, stringsAsFactors = FALSE, check.names = FALSE)
}

# region 시계열 한 변수 보간: 내부 선형 + 양끝 ≤max_carry 연장
.interp_vec <- function(years, v, max_carry) {
  o <- order(years); yy <- years[o]; vv <- as.numeric(v[o])
  ok <- which(is.finite(vv))
  if (length(ok) >= 2) vv <- approx(yy[ok], vv[ok], xout = yy, rule = 1)$y   # 내부 선형, 양끝 NA
  ok2 <- which(is.finite(vv))
  if (length(ok2) > 0) {
    f <- min(ok2); l <- max(ok2)
    if (f > 1)            for (i in (f - 1):1)            { if (f - i <= max_carry) vv[i] <- vv[f] else break }
    if (l < length(vv))   for (i in (l + 1):length(vv))   { if (i - l <= max_carry) vv[i] <- vv[l] else break }
  }
  out <- rep(NA_real_, length(v)); out[o] <- vv; out
}

# 병합 후 1회: 스팬 게이트(드롭) + region별 다년 보간
finalize_covariates <- function(cm, ys, ye) {
  meta <- c("region","year","cases","population","rate_100k","disease","area_2","area_3")
  cov_cols <- setdiff(names(cm), meta)
  cov_cols <- cov_cols[vapply(cm[cov_cols], is.numeric, logical(1))]
  if (length(cov_cols) == 0) return(cm)

  # ── (0) 결측-0 코딩 정리: '한 해 전체가 0'인 칸은 결측(NA)으로 (KCHS 설문 미조사연도 등) ──
  #     실제 공간변수가 229개 시군구 전부 정확히 0일 리 없음 → 0=미조사 코드로 판단.
  #     이후 스팬게이트가 이런 변수(마스크·코로나·거리두기 = 2020만 실측)를 자동 드롭.
  cm <- cm %>% group_by(year) %>%
    mutate(across(all_of(cov_cols),
      ~ if (any(is.finite(.x)) && all(.x[is.finite(.x)] == 0)) rep(NA_real_, length(.x)) else .x)) %>%
    ungroup()

  lo_edge <- ys + MAX_CARRY; hi_edge <- ye - MAX_CARRY
  drop_cols <- c()
  for (c in cov_cols) {
    ry <- unique(cm$year[is.finite(cm[[c]])])
    if (length(ry) == 0 || min(ry) > lo_edge || max(ry) < hi_edge) drop_cols <- c(drop_cols, c)
  }
  if (length(drop_cols) > 0) {
    cm[drop_cols] <- NULL
    cat(sprintf("    · 스팬게이트 제외 %d개 (실측이 %d~%d 양끝 미커버):\n      %s\n",
        length(drop_cols), lo_edge, hi_edge, paste(drop_cols, collapse=", ")))
  }
  keep_cols <- setdiff(cov_cols, drop_cols)
  if (length(keep_cols) > 0)
    cm <- cm %>% arrange(region, year) %>% group_by(region) %>%
      mutate(across(all_of(keep_cols), ~ .interp_vec(year, .x, MAX_CARRY))) %>% ungroup()
  cm
}

build_cor_merged <- function(disease, ys, ye, include_food = FALSE) {
  # ── 종속 ──
  df_target <- DF_RAW_CACHE %>%
    filter(disease == !!disease, year >= ys, year <= ye) %>%
    clean_region(ys, ye) %>%
    group_by(region, year) %>%
    summarise(cases = sum(cases, na.rm = TRUE),
              population = mean(population, na.rm = TRUE), .groups = "drop") %>%
    mutate(rate_100k = cases / population * 100000)
  cm <- df_target

  # ── (1) 건강지표 parquet (raw 병합; 보간은 finalize에서) ──
  tryCatch({
    hpq <- read_parquet(PATH_HEALTH_PQ) %>% as.data.frame() %>% clean_region(ys, ye)
    ah <- intersect(HEALTH_VARS, names(hpq))
    for (v in ah) hpq[[v]] <- suppressWarnings(as.numeric(hpq[[v]]))
    hagg <- hpq %>% group_by(region, year) %>% summarise(across(all_of(ah), ~mean(.x, na.rm = TRUE)), .groups = "drop")
    cm <- cm %>% left_join(hagg, by = c("region","year"))
  }, error = function(e) cat(sprintf("    ❌ parquet: %s\n", e$message)))

  # ── (2) 일반 CSV (raw 병합) ──
  load_list <- function(cm, lst) {
    for (fn in names(lst)) {
      fp <- file.path(BASE_IV, fn); if (!file.exists(fp)) next
      raw <- read_csv_safe(fp); if (is.null(raw)) next
      raw <- raw %>% clean_region(ys, ye)
      av <- intersect(lst[[fn]], names(raw)); av <- setdiff(av, names(cm))
      if (length(av) == 0) next
      for (v in av) raw[[v]] <- suppressWarnings(as.numeric(raw[[v]]))
      agg <- raw %>% group_by(region, year) %>% summarise(across(all_of(av), ~mean(.x, na.rm = TRUE)), .groups = "drop")
      cm <- cm %>% left_join(agg, by = c("region","year"))
    }
    cm
  }
  cm <- load_list(cm, SELECTED)
  cm <- load_list(cm, SELECTED_NEW)

  # ── (3) 식품매개 (민감도 모드에서만) ──
  if (include_food) {
    cm <- load_list(cm, FOOD_FILES_SELECTED)
    add_shellfish <- function(cm, fn, prefix, patterns) {
      fp <- file.path(BASE_IV, fn); if (!file.exists(fp)) return(cm)
      raw <- tryCatch(read_csv_safe(fp) %>% clean_region(ys, ye), error = function(e) NULL)
      if (is.null(raw)) return(cm)
      meta_c <- c("region","year","sido","sigungu","X")
      num_c <- setdiff(names(raw), meta_c)
      for (v in num_c) raw[[v]] <- suppressWarnings(as.numeric(raw[[v]]))
      newv <- c()
      for (nm in names(patterns)) {
        cols <- num_c[grepl(patterns[[nm]], num_c)]
        if (length(cols) > 0) { cn <- paste0(prefix, "_", nm); raw[[cn]] <- rowSums(raw[cols], na.rm = TRUE); newv <- c(newv, cn) }
      }
      newv <- setdiff(newv, names(cm)); if (length(newv) == 0) return(cm)
      agg <- raw %>% group_by(region, year) %>% summarise(across(all_of(newv), ~mean(.x, na.rm = TRUE)), .groups = "drop")
      cm %>% left_join(agg, by = c("region","year"))
    }
    cm <- add_shellfish(cm, "어패류_패류_전처리.csv", "패류",
            list(자연채묘총합="자연채묘", 인공종묘총합="인공종묘", 전체총합="(자연채묘|인공종묘|인공종자)"))
    cm <- add_shellfish(cm, "어패류_갑각류_전처리.csv", "갑각류",
            list(전체총합=".", 새우류총합="새우|대하", 게류총합="게|꽃게"))
    cm <- add_shellfish(cm, "어패류_어류_전처리.csv", "어류",
            list(수정란총합="수정란", 치어총합="치어"))
    cm <- add_shellfish(cm, "어패류_해조류_김_전처리.csv", "해조류", list(김총합="."))
    cm <- add_shellfish(cm, "어패류_해조류_김제외_전처리.csv", "해조류", list(기타총합="."))
  }

  # ── (4) 스팬 게이트 + 다년 보간 ──
  cm <- finalize_covariates(cm, ys, ye)

  # ── 도시/농촌 분류 ──
  cm$area_3 <- sapply(cm$region, classify_region)
  cm$area_2 <- ifelse(cm$area_3 == "농촌", "농촌", "도시")
  cm
}

cat("✅ 01_데이터로드_함수.R 로드 완료 (2011–2024 다년보간 + 스팬게이트)\n")
