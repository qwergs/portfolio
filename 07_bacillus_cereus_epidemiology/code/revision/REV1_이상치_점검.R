# ==============================================================================
#  REV1 자료품질 : 통합시 시-구 집계 오류 전수 점검
#  ------------------------------------------------------------------------
#  2020년 도시지역인구비율 10건(전부 통합시, 값이 자치구 수만큼 부풀음)과
#  2024년 의사수 12건(전부 통합시, 0으로 기록)이 같은 실패 유형입니다.
#  나머지 19개 사전지정 변수에도 같은 서명이 있는지 전수 점검합니다.
#
#  실행:  Rscript -e 'source("REV1_이상치_점검.R", encoding="UTF-8")'
#  (INLA 적합 없음. 1분 내외)
# ==============================================================================

rm(list = ls()); gc()

# ── 0. 경로 + 소스 로드 (원본 스크립트와 동일) ────────────────────────────────
ROOT     <- "/Users/guns/Desktop/과거자료/학부생_전처리_논문/도시_농촌_분석"
OLD_ROOT <- "/Users/guns/Desktop/도시_농촌_분석"
OUTDIR   <- file.path(ROOT, "B_cereus", "IJEHR_REV1")
if (!dir.exists(ROOT)) stop("ROOT 경로가 없습니다: ", ROOT)
if (!dir.exists(OUTDIR)) dir.create(OUTDIR, recursive = TRUE)

# ★ 2026-09-15: 투고본(5변환) 코드로 고정.
#   ROOT/code/02_모델_함수.R 은 2026-07-09 에 5변환 → 8변환으로 개정되었고
#   성비 raw 특례·사육두수 per10k 특례가 제거되었습니다(파일 mtime·헤더 주석으로 확인).
#   투고 원고(MAIN 260620_2016.xlsx, 2026-06-23)는 개정 이전 코드로 산출되었으므로,
#   리비전 민감도 분석은 반드시 투고 당시 코드 = ZENODO 기탁본으로 적합해야 합니다.
#   FALSE 로 두면 현행(8변환) 코드를 씁니다. 원고 재현에는 TRUE 가 맞습니다.
USE_SUBMITTED_CODE <- TRUE
CODEDIR  <- if (USE_SUBMITTED_CODE) {
  file.path(ROOT, "B_cereus", "기타", "ZENODO_DEPOSIT", "code")
} else {
  file.path(ROOT, "code")
}
OLD_DATA <- "/Users/guns/Desktop/건상_시각화/FBD_DATA_ZIP"   # 기탁본에 박힌 옛 데이터 경로
NEW_DATA <- file.path(ROOT, "FBD_DATA_ZIP")
if (!dir.exists(CODEDIR)) stop("CODEDIR 없음: ", CODEDIR)
if (!dir.exists(NEW_DATA)) stop("데이터 폴더 없음: ", NEW_DATA)
cat(sprintf("  · 코드 출처: %s\n", ifelse(USE_SUBMITTED_CODE, "ZENODO 기탁본 (투고본 5변환)", "현행 작업본 (8변환)")))

src_patched <- function(fname) {
  fp <- file.path(CODEDIR, fname)
  if (!file.exists(fp)) stop("소스 없음: ", fp)
  txt <- readLines(fp, encoding = "UTF-8", warn = FALSE)
  txt <- gsub(OLD_DATA, NEW_DATA, txt, fixed = TRUE)
  txt <- gsub(OLD_ROOT, ROOT,     txt, fixed = TRUE)
  eval(parse(text = paste(txt, collapse = "\n")), envir = globalenv())
  cat(sprintf("  · source: %s\n", fname))
}
cat("■ 0. 소스 로드\n")
src_patched("00_config_공통.R")
src_patched("01_데이터로드_함수.R")
src_patched("02_모델_함수.R")

DZ <- "바실러스세레우스"
YS <- 2011; YE <- 2024
TS <- format(Sys.time(), "%y%m%d_%H%M")
NONSPATIAL       <- "+ f(idtime,model='rw1',hyper=pc_prec) + f(idarea,model='iid',hyper=pc_prec)"
PRINCIPAL_FAMILY <- "zeroinflatednbinomial1"
pc_prec <- list(prec = list(prior = "pc.prec", param = c(0.5, 0.01)))


# ── 1. 공간객체 (원본 스크립트 §1 그대로) (주분석과 동일) ───────────────────────────────────────────────
cat("\n■ 1. 공간객체\n")
shp <- sf::st_read(PATH_SHP, quiet = TRUE) %>%
  mutate(region = stringr::str_replace_all(as.character(region), "\\s+", ""),
         region = dplyr::if_else(region == "인천시미추홀구", "인천시남구", region))
EXCLUDED_ISLANDS <- c("인천시옹진군","전라남도완도군","전라남도진도군",
                      "경상남도거제시","경상남도남해군","경상북도울릉군")
shp_main <- shp %>% filter(!region %in% EXCLUDED_ISLANDS)
nb_obj <- spdep::poly2nb(shp_main, snap = 0.01)
iso <- which(spdep::card(nb_obj) == 0)
ISO_REGIONS <- if (length(iso) > 0) shp_main$region[iso] else character(0)
if (length(iso) > 0) { shp_main <- shp_main[-iso, ]; nb_obj <- spdep::poly2nb(shp_main, snap = 0.01) }
spdep::nb2INLA(nb_obj, file = file.path(tempdir(), "bc_all.graph"))
g_main <- INLA::inla.read.graph(file.path(tempdir(), "bc_all.graph"))
nb_w <- spdep::nb2listw(nb_obj, style = "W")
cat(sprintf("  모형 대상 시군구: %d개 (전체 %d개 중 도서·비인접 %d개 제외)\n",
            nrow(shp_main), nrow(shp), nrow(shp) - nrow(shp_main)))
cat(sprintf("  제외: %s\n", paste(c(EXCLUDED_ISLANDS, ISO_REGIONS), collapse = ", ")))

# ── 1b. 패널 생성 (PRE_RAW 후킹) ───────────────────────────────────────────────
cat("\n■ 1b. 패널 생성\n")
CUR_YS <<- YS; CUR_YE <<- YE
PRE_RAW <- NULL
.orig_finalize <- finalize_covariates
finalize_covariates <- function(cm, ys, ye) { PRE_RAW <<- cm; .orig_finalize(cm, ys, ye) }
cm <- build_cor_merged(DZ, YS, YE, include_food = FALSE)
finalize_covariates <- .orig_finalize
stopifnot(!is.null(PRE_RAW))




# ── 2. 통합시 집계 오류 전수 점검 ───────────────────────────────────────────
#  서명: 같은 지역의 다른 연도와 동떨어진 단일 연도 값.
#  2020년 도시지역인구비율(10건), 2024년 의사수(12건)에서 이미 확인된 패턴.
cat("\n■ 2. 단일 연도 이상치 전수 점검\n")
stopifnot(!is.null(PRE_RAW))
VARS <- intersect(TV_PRESPEC$code, names(PRE_RAW))
cat(sprintf("  점검 대상: 사전지정 %d개 변수 × %d개 지역 × %d년\n",
            length(VARS), dplyr::n_distinct(PRE_RAW$region), length(YS:YE)))

# 자치구를 둔 통합시 (구 단위 원자료가 시 단위로 합산되며 오류가 나는 지역)
CONSOL <- c("고양시","부천시","성남시","수원시","안산시","안양시","용인시",
            "창원시","포항시","전주시","천안시","청주시")
is_consol <- function(r) any(sapply(CONSOL, function(p) grepl(p, r, fixed = TRUE)))

out <- data.frame()
for (v in VARS) {
  d <- data.frame(region = PRE_RAW$region, year = PRE_RAW$year,
                  x = suppressWarnings(as.numeric(PRE_RAW[[v]])))
  d <- d[is.finite(d$x), ]
  if (!nrow(d)) next
  g <- d %>% dplyr::group_by(region) %>%
    dplyr::mutate(med = median(x, na.rm = TRUE),
                  mad = mad(x, na.rm = TRUE),
                  n_obs = dplyr::n()) %>% dplyr::ungroup()
  # 지역 중앙값 대비 배수 + robust z
  g$ratio <- ifelse(g$med > 0, g$x / g$med, NA_real_)
  g$rz    <- ifelse(g$mad > 0, abs(g$x - g$med) / g$mad, NA_real_)
  flag <- g %>% dplyr::filter(n_obs >= 5,
                              (is.finite(ratio) & (ratio > 1.5 | ratio < 1/1.5)) |
                              (is.finite(rz) & rz > 10))
  if (nrow(flag)) {
    flag$variable <- v
    flag$consolidated_city <- sapply(flag$region, is_consol)
    out <- dplyr::bind_rows(out, flag)
  }
}
if (!nrow(out)) { cat("  이상치 없음\n") } else {
  out <- out %>% dplyr::select(variable, region, year, value = x,
                               region_median = med, ratio, robust_z = rz,
                               consolidated_city) %>%
    dplyr::arrange(variable, dplyr::desc(ratio))
  cat(sprintf("  ★ 플래그 %d건 (변수 %d개)\n", nrow(out), dplyr::n_distinct(out$variable)))

  cat("\n  [연도별 집중도] 특정 연도에 몰리면 집계 오류 의심\n")
  print(as.data.frame(out %>% dplyr::count(year, name = "n") %>%
                        dplyr::arrange(dplyr::desc(n))), row.names = FALSE)

  cat("\n  [변수 × 통합시 여부]\n")
  print(as.data.frame(out %>% dplyr::count(variable, consolidated_city, name = "n") %>%
                        dplyr::arrange(dplyr::desc(n))), row.names = FALSE)

  cat("\n  [통합시에 집중된 변수-연도 조합 — 최우선 확인 대상]\n")
  hot <- out %>% dplyr::group_by(variable, year) %>%
    dplyr::summarise(n = dplyr::n(), n_consol = sum(consolidated_city), .groups = "drop") %>%
    dplyr::filter(n_consol >= 3) %>% dplyr::arrange(dplyr::desc(n_consol))
  if (nrow(hot)) print(as.data.frame(hot), row.names = FALSE) else cat("    없음\n")
}

TS2 <- format(Sys.time(), "%y%m%d_%H%M")
xp <- file.path(OUTDIR, sprintf("REV1_이상치점검_%s.xlsx", TS2))
wb <- openxlsx::createWorkbook()
openxlsx::addWorksheet(wb, "29_outlier_scan")
openxlsx::writeData(wb, "29_outlier_scan", if (nrow(out)) out else data.frame(note = "no outliers"))
openxlsx::saveWorkbook(wb, xp, overwrite = TRUE)
cat(sprintf("\n  ✔ %s\n", xp))
cat("\n※ '연도별 집중도'에서 한 해에 몰린 항목, 그리고 '통합시에 집중된 조합'을\n")
cat("   먼저 보십시오. 그것이 시-구 집계 정합성 오류의 서명입니다.\n")
