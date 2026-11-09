# ==============================================================================
# REV1_추가분석.R — IJEHR B. cereus 리비전 1차 추가분석 (2026-09-14)
#
#   목적: 심사 지적 중 "재적합이 필요한 항목"을 한 번에 처리
#     [N1 ] R2-14  영과잉 이항부(0건 vs 1건 이상) 로지스틱 모형 (전체/도시/농촌)
#     [N1b] R2-14  허들(zeroinflatednbinomial0) 모형 병행
#     [N2 ] R2-10  비표준화(원 변환단위) IRR 병기용 재적합
#     [N3 ] §4-A   형태 고정 + 스크리닝 없음 민감도 (가장 중요: 결론 유지 여부 판정)
#     [N4 ] R2-9   결측·보간 감사 (변수 × 연도 × 상태)
#     [N5 ] R2-10  공변량 기술통계 + 채택 형태 + 절단값
#     [N6 ] R2-12  사전분포 전수 추출 + PC prior 민감도(u = 0.1 / 0.5 / 1.0)
#     [N7 ] R2-13  ZINB 영과잉 확률 사후추정 + 환경정보(INLA/R 버전, 실행시간)
#
#   실행:  source("REV1_추가분석.R", encoding = "UTF-8")
#   출력:  <B_cereus>/IJEHR_REV1/REV1_추가분석_<타임스탬프>.xlsx  (+ .rds, sessionInfo.txt)
#   소요:  INLA 적합 약 15~25회. 대략 10~30분 예상.
#
#   ※ 주분석 코드(BC_도농_통합분석.R)는 건드리지 않습니다. 읽기만 합니다.
#   ※ 경로가 옮겨졌으므로 원본 .R 파일의 하드코딩 경로를 메모리상에서만 치환해 source 합니다.
# ==============================================================================

rm(list = ls()); gc()
T_START <- Sys.time()

# ── 0. 경로 ───────────────────────────────────────────────────────────────────
ROOT     <- "/Users/guns/Desktop/과거자료/학부생_전처리_논문/도시_농촌_분석"  # 현재 위치
OLD_ROOT <- "/Users/guns/Desktop/도시_농촌_분석"                              # 코드에 박힌 옛 경로
OUTDIR   <- file.path(ROOT, "B_cereus", "IJEHR_REV1")
if (!dir.exists(ROOT)) stop("ROOT 경로가 없습니다. 실제 폴더 위치로 ROOT를 고치세요: ", ROOT)
if (!dir.exists(OUTDIR)) dir.create(OUTDIR, recursive = TRUE)

# 원본 소스를 읽어 경로만 치환한 뒤 평가 (원본 파일은 수정하지 않음)
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
NONSPATIAL <- "+ f(idtime,model='rw1',hyper=pc_prec) + f(idarea,model='iid',hyper=pc_prec)"
PRINCIPAL_FAMILY <- "zeroinflatednbinomial1"

pc_bym  <- list(prec.unstruct = list(prior = "pc.prec", param = c(0.5, 0.01)),
                prec.spatial  = list(prior = "pc.prec", param = c(0.5, 0.01)))
pc_prec <- list(prec = list(prior = "pc.prec", param = c(0.5, 0.01)))

fit_inla <- function(fs, dat, fam = PRINCIPAL_FAMILY, ...) {
  tryCatch(inla(as.formula(fs), family = fam, data = dat,
                control.compute = list(dic = TRUE, waic = TRUE, cpo = TRUE),
                control.predictor = list(link = 1), verbose = FALSE, ...),
           error = function(e) { cat(sprintf("    ❌ fit 실패: %s\n", e$message)); NULL })
}
irr_tbl <- function(fit, drop_intercept = TRUE, label = "IRR") {
  if (is.null(fit)) return(data.frame())
  fe <- fit$summary.fixed
  if (drop_intercept) fe <- fe[rownames(fe) != "(Intercept)", , drop = FALSE]
  out <- data.frame(term = rownames(fe),
                    est = exp(fe$mean), lo = exp(fe$`0.025quant`), hi = exp(fe$`0.975quant`),
                    sig = ifelse(fe$`0.025quant` > 0 | fe$`0.975quant` < 0, "*", ""),
                    stringsAsFactors = FALSE)
  names(out)[2:4] <- c(label, paste0(label, "_lo"), paste0(label, "_hi"))
  out
}

# ── 1. 공간객체 (주분석과 동일) ───────────────────────────────────────────────
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

# ── 2. 패널 생성 + [N4] 결측·보간 감사 ────────────────────────────────────────
cat("\n■ 2. 패널 생성 + 결측·보간 감사\n")
CUR_YS <<- YS; CUR_YE <<- YE

PRE_RAW <- NULL; POST_FIN <- NULL
.orig_finalize <- finalize_covariates
finalize_covariates <- function(cm, ys, ye) {          # 감사용 래퍼
  PRE_RAW  <<- cm
  out <- .orig_finalize(cm, ys, ye)
  POST_FIN <<- out
  out
}
cm <- build_cor_merged(DZ, YS, YE, include_food = FALSE)
finalize_covariates <- .orig_finalize                   # 원복

audit <- data.frame()
if (!is.null(PRE_RAW) && !is.null(POST_FIN)) {
  meta <- c("region","year","cases","population","rate_100k","disease","area_2","area_3")
  vcols <- intersect(setdiff(names(POST_FIN), meta), names(PRE_RAW))
  for (v in vcols) {
    pre  <- PRE_RAW[, c("region","year",v)];  names(pre)[3]  <- "pre"
    post <- POST_FIN[, c("region","year",v)]; names(post)[3] <- "post"
    d <- dplyr::left_join(post, pre, by = c("region","year"))
    # 연도 전체가 0 → 미조사 코드로 간주되어 NA 처리된 케이스 식별
    zeroyr <- d %>% group_by(year) %>%
      summarise(all_zero = any(is.finite(pre)) && all(pre[is.finite(pre)] == 0), .groups = "drop")
    d <- dplyr::left_join(d, zeroyr, by = "year")
    # 지역별 최초/최후 실측연도 기준으로 내부보간 vs 양끝연장 구분
    d <- d %>% group_by(region) %>%
      mutate(first_obs = suppressWarnings(min(year[is.finite(pre) & !all_zero])),
             last_obs  = suppressWarnings(max(year[is.finite(pre) & !all_zero]))) %>% ungroup()
    d$status <- with(d, dplyr::case_when(
      is.finite(pre) & !all_zero                        ~ "observed",
      all_zero                                          ~ "recoded_zero_year_to_NA",
      !is.finite(post)                                  ~ "missing",
      is.finite(post) & year > first_obs & year < last_obs ~ "interpolated",
      is.finite(post)                                   ~ "carried_forward_backward",
      TRUE                                              ~ "missing"))
    tab <- d %>% count(year, status) %>% tidyr::pivot_wider(names_from = status, values_from = n, values_fill = 0)
    tab$variable <- v
    audit <- dplyr::bind_rows(audit, tab)
  }
  audit <- audit %>% dplyr::relocate(variable, year)
  cat(sprintf("  감사 대상 변수 %d개 × 연도 → %d행\n", length(vcols), nrow(audit)))
}

# ── 2b. [N0] ★ 결측-0 코딩 진단 및 교정 패널 ─────────────────────────────────
#   문제: 파이프라인의 "0 → NA" 규칙은 '한 해 229개 지역이 전부 0'일 때만 작동.
#         일부 지역만 0인 결측-0 코딩은 그대로 0(=보급률 0%, 진료비 0원)으로 투입됨.
#   실제 확인된 사례(원자료 점검, 2026-09-14):
#     · 상수도보급률       2018–2024 각 연도 76개 지역(주로 광역시 자치구)이 0
#     · 관내진료비_전체    2023년 85개 지역이 0
#     · 의사수             2024년 12개 통합시(구 있는 시)가 0
#   → 0이 물리적으로 불가능한 변수는 0을 NA로 재코딩한 뒤 동일 보간 로직 재적용.
cat("\n■ 2b. [N0] 결측-0 코딩 진단 + 교정 패널 ★\n")
ZERO_INVALID <- c("상수도보급률","하수도보급률","의사수","관내진료비_전체",
                  "연간인플루엔자예방접종률_표준화율","건강생활실천율_조율",
                  "걷기실천율_표준화율","재정자립도","재정자주도","성비","고령인구비율")
# ※ 0이 실제로 가능한 변수(토지이용 면적, 정화조인구, 지하수 개소수, 고도처리율,
#    도시지역인구비율)는 의도적으로 제외했습니다. 필요하면 위 목록을 조정하십시오.

zero_diag <- data.frame()
if (!is.null(PRE_RAW)) {
  for (v in intersect(TV_PRESPEC$code, names(PRE_RAW))) {
    x <- suppressWarnings(as.numeric(PRE_RAW[[v]]))
    d <- data.frame(year = PRE_RAW$year, x = x)
    t <- d %>% group_by(year) %>%
      summarise(n_obs = sum(is.finite(x)), n_zero = sum(is.finite(x) & x == 0),
                pct_zero = round(100 * sum(is.finite(x) & x == 0) / max(1, sum(is.finite(x))), 1),
                all_zero_year = n_obs > 0 && n_zero == n_obs, .groups = "drop")
    t$variable <- v; t$flagged_zero_invalid <- v %in% ZERO_INVALID
    zero_diag <- dplyr::bind_rows(zero_diag, t)
  }
  zero_diag <- zero_diag %>% dplyr::relocate(variable, year)
  bad <- zero_diag %>% filter(flagged_zero_invalid, n_zero > 0, !all_zero_year)
  cat(sprintf("  0-코딩 의심 (변수×연도) %d건:\n", nrow(bad)))
  if (nrow(bad) > 0) print(head(bad %>% arrange(desc(n_zero)), 20), row.names = FALSE)
}

PRE_FIX <- PRE_RAW
n_recoded <- 0
for (v in intersect(ZERO_INVALID, names(PRE_FIX))) {
  x <- suppressWarnings(as.numeric(PRE_FIX[[v]]))
  k <- sum(is.finite(x) & x == 0); n_recoded <- n_recoded + k
  x[is.finite(x) & x == 0] <- NA_real_
  PRE_FIX[[v]] <- x
}
cat(sprintf("  0 → NA 재코딩: %d 셀\n", n_recoded))
cm_fix <- .orig_finalize(PRE_FIX, YS, YE)
cm_fix$area_3 <- sapply(cm_fix$region, classify_region)
cm_fix$area_2 <- ifelse(cm_fix$area_3 == "농촌", "농촌", "도시")
dropped_fix <- setdiff(names(cm), names(cm_fix))
cat(sprintf("  교정 후 스팬게이트로 추가 탈락한 변수: %s\n",
            ifelse(length(dropped_fix) == 0, "없음", paste(dropped_fix, collapse = ", "))))

# ── 3. 주모형 재현 (Table 2와 동일 절차) ──────────────────────────────────────
cat("\n■ 3. 주모형 재현 (ZINB, 비공간 M0)\n")
DAT_all   <- cm
DAT_urban <- cm %>% filter(area_2 == "도시")
DAT_rural <- cm %>% filter(area_2 == "농촌")
res_all   <- run_model(TV_PRESPEC, DAT_all,   quiet = TRUE, re_str = NONSPATIAL, fam = PRINCIPAL_FAMILY)
res_urban <- run_model(TV_PRESPEC, DAT_urban, quiet = TRUE, re_str = NONSPATIAL, fam = PRINCIPAL_FAMILY)
res_rural <- run_model(TV_PRESPEC, DAT_rural, quiet = TRUE, re_str = NONSPATIAL, fam = PRINCIPAL_FAMILY)
stopifnot(!is.null(res_all$ic), !is.null(res_all$FMAP))
cat(sprintf("  전체 N=%d / 도시 N=%d / 농촌 N=%d\n",
            nrow(res_all$ic), nrow(res_urban$ic), nrow(res_rural$ic)))
cat("  ※ 이 N이 원고의 3,122와 일치하는지 반드시 확인하십시오.\n")

STRATA <- list(Overall = res_all, Urban = res_urban, Rural = res_rural)

# ── 3b. [N0] 교정 패널로 주모형 재적합 ★ 결론 유지 여부 판정 ─────────────────
cat("\n■ 3b. [N0] 교정 패널 재적합 ★\n")
fix_all   <- run_model(TV_PRESPEC, cm_fix,                              quiet = TRUE, re_str = NONSPATIAL, fam = PRINCIPAL_FAMILY)
fix_urban <- run_model(TV_PRESPEC, cm_fix %>% filter(area_2 == "도시"), quiet = TRUE, re_str = NONSPATIAL, fam = PRINCIPAL_FAMILY)
fix_rural <- run_model(TV_PRESPEC, cm_fix %>% filter(area_2 == "농촌"), quiet = TRUE, re_str = NONSPATIAL, fam = PRINCIPAL_FAMILY)
corrected_irr <- dplyr::bind_rows(
  { d <- fix_all$mv;   if (!is.null(d) && nrow(d)) { d$stratum <- "Overall"; d } else data.frame() },
  { d <- fix_urban$mv; if (!is.null(d) && nrow(d)) { d$stratum <- "Urban";   d } else data.frame() },
  { d <- fix_rural$mv; if (!is.null(d) && nrow(d)) { d$stratum <- "Rural";   d } else data.frame() })
for (nm in c("Overall","Urban","Rural")) {
  R0 <- STRATA[[nm]]; R1 <- switch(nm, Overall = fix_all, Urban = fix_urban, Rural = fix_rural)
  cat(sprintf("  %s: 원본 N=%s 유의=%s  →  교정 N=%s 유의=%s\n", nm,
              ifelse(is.null(R0$ic), NA, nrow(R0$ic)), R0$sig_count,
              ifelse(is.null(R1$ic), NA, nrow(R1$ic)), R1$sig_count))
}
# 원본 vs 교정 IRR 대조표
cmp_fix <- data.frame()
for (nm in c("Overall","Urban","Rural")) {
  a <- STRATA[[nm]]$mv; b <- switch(nm, Overall = fix_all, Urban = fix_urban, Rural = fix_rural)$mv
  if (is.null(a) || is.null(b) || !nrow(a) || !nrow(b)) next
  m <- merge(a[, c("var_kr","형태","IRR","lo","hi","sig")],
             b[, c("var_kr","형태","IRR","lo","hi","sig")],
             by = "var_kr", all = TRUE, suffixes = c("_orig","_fix"))
  m$stratum <- nm
  m$flip <- with(m, ifelse(is.na(sig_orig) | is.na(sig_fix), "only_one_model",
                    ifelse(sig_orig == sig_fix, "", "SIGNIFICANCE_CHANGED")))
  cmp_fix <- dplyr::bind_rows(cmp_fix, m)
}
if (nrow(cmp_fix)) {
  chg <- cmp_fix %>% filter(flip != "")
  cat(sprintf("  ★ 유의성이 바뀐 항목: %d건\n", nrow(chg)))
  if (nrow(chg)) print(chg[, c("stratum","var_kr","IRR_orig","sig_orig","IRR_fix","sig_fix","flip")], row.names = FALSE)
}

# ── 4. [N7] ZINB 영과잉 확률 사후 + [N6] 사전분포 전수 추출 ───────────────────
cat("\n■ 4. 영과잉 확률 사후 + 사전분포 추출\n")
hyper_post <- data.frame(); prior_spec <- data.frame()
for (nm in names(STRATA)) {
  f <- STRATA[[nm]]$fit
  if (is.null(f)) next
  hp <- f$summary.hyperpar
  hyper_post <- dplyr::bind_rows(hyper_post,
    data.frame(stratum = nm, parameter = rownames(hp), hp, row.names = NULL, check.names = FALSE))
  ah <- tryCatch(f$all.hyper, error = function(e) NULL)
  if (!is.null(ah)) {
    txt <- paste(capture.output(str(ah, max.level = 4)), collapse = "\n")
    prior_spec <- dplyr::bind_rows(prior_spec, data.frame(stratum = nm, all_hyper_dump = txt))
  }
}
print(hyper_post[, 1:5], row.names = FALSE)

# ── 5. [N1] 영과잉 이항부: 0건 vs 1건 이상 ────────────────────────────────────
cat("\n■ 5. [N1] 이항부 로지스틱 (any case vs zero)\n")
bin_out <- data.frame(); bin_fit_list <- list()
for (nm in names(STRATA)) {
  R <- STRATA[[nm]]; if (is.null(R$ic)) next
  ic <- R$ic
  ic$any    <- as.integer(ic$cases > 0)
  ic$logpop <- as.numeric(scale(log(ic$population + 1)))   # 인구규모 보정(필수)
  cov_str <- paste(R$FMAP$safe, collapse = " + ")
  fs <- paste("any ~", cov_str, "+ logpop", NONSPATIAL)
  fb <- tryCatch(inla(as.formula(fs), family = "binomial", Ntrials = 1, data = ic,
                      control.compute = list(dic = TRUE, waic = TRUE, cpo = TRUE),
                      control.predictor = list(link = 1)), error = function(e) NULL)
  bin_fit_list[[nm]] <- fb
  if (!is.null(fb)) {
    t <- irr_tbl(fb, label = "OR"); t$stratum <- nm
    t$variable_kr <- vapply(t$term, function(z) {
      rn <- sub("_z$", "", z); k <- R$FMAP$kr[R$FMAP$eng == rn]
      if (length(k)) k[1] else rn }, character(1))
    t$zero_rate <- mean(ic$any == 0)
    t$DIC <- fb$dic$dic
    bin_out <- dplyr::bind_rows(bin_out, t)
    cat(sprintf("  %s: N=%d, 0건 비율=%.3f, DIC=%.1f, 유의 변수 %d개\n",
                nm, nrow(ic), mean(ic$any == 0), fb$dic$dic, sum(t$sig == "*")))
  }
}

# ── 6. [N1b] 허들 모형 ────────────────────────────────────────────────────────
cat("\n■ 6. [N1b] 허들 모형 (zeroinflatednbinomial0)\n")
hurdle_out <- data.frame()
for (nm in names(STRATA)) {
  R <- STRATA[[nm]]; if (is.null(R$ic)) next
  fs <- paste("cases ~", paste(R$FMAP$safe, collapse = " + "),
              "+ offset(log(population+1))", NONSPATIAL)
  fh <- fit_inla(fs, R$ic, fam = "zeroinflatednbinomial0")
  if (!is.null(fh)) {
    t <- irr_tbl(fh, label = "IRR_hurdle"); t$stratum <- nm; t$DIC <- fh$dic$dic
    hurdle_out <- dplyr::bind_rows(hurdle_out, t)
    cat(sprintf("  %s: DIC=%.1f (ZINB1 DIC=%.1f)\n", nm, fh$dic$dic,
                ifelse(is.null(R$fit), NA, R$fit$dic$dic)))
  }
}

# ── 7. [N2] 비표준화 IRR (원 변환단위) ────────────────────────────────────────
cat("\n■ 7. [N2] 비표준화 IRR\n")
unstd_out <- data.frame()
for (nm in names(STRATA)) {
  R <- STRATA[[nm]]; if (is.null(R$ic) || is.null(R$data_ext)) next
  ic <- R$ic
  uvars <- c()
  for (i in seq_len(nrow(R$FMAP))) {
    src <- R$FMAP$code[i]; dst <- paste0(R$FMAP$eng[i], "_u")
    if (!src %in% names(ic)) next
    ic[[dst]] <- as.numeric(ic[[src]]); uvars <- c(uvars, dst)
  }
  if (length(uvars) == 0) next
  fs <- paste("cases ~", paste(uvars, collapse = " + "),
              "+ offset(log(population+1))", NONSPATIAL)
  fu <- fit_inla(fs, ic, fam = PRINCIPAL_FAMILY)
  if (!is.null(fu)) {
    t <- irr_tbl(fu, label = "IRR_unstd"); t$stratum <- nm
    t$form <- R$FMAP$형태[match(sub("_u$", "", t$term), R$FMAP$eng)]
    t$variable_kr <- R$FMAP$kr[match(sub("_u$", "", t$term), R$FMAP$eng)]
    unstd_out <- dplyr::bind_rows(unstd_out, t)
  }
}

# ── 8. [N5] 공변량 기술통계 + 채택 형태 + 절단값 ──────────────────────────────
cat("\n■ 8. [N5] 공변량 사전(기술통계·형태·절단값)\n")
covdict <- data.frame()
for (nm in names(STRATA)) {
  R <- STRATA[[nm]]; if (is.null(R$form_map)) next
  DAT <- if (nm == "Overall") DAT_all else if (nm == "Urban") DAT_urban else DAT_rural
  for (v in names(R$form_map)) {
    m <- R$form_map[[v]]; x <- suppressWarnings(as.numeric(DAT[[v]]))
    xf <- x[is.finite(x)]
    cut_txt <- ""
    if (m$형태 == "binary") {
      zp <- sum(xf == 0) / length(xf) * 100
      cut_txt <- if (zp > 20) "x > 0" else sprintf("x > median = %.4g", median(xf))
    } else if (m$형태 == "T3") {
      zp <- sum(xf == 0) / length(xf) * 100
      cut_txt <- if (zp > 20) sprintf("0 | 0<x<=%.4g | x>%.4g", median(xf[xf>0]), median(xf[xf>0]))
                 else sprintf("tertile cuts: %.4g, %.4g", quantile(xf,1/3), quantile(xf,2/3))
    } else if (m$형태 == "Q4") {
      cut_txt <- sprintf("quartile cuts: %.4g, %.4g, %.4g",
                         quantile(xf,.25), quantile(xf,.5), quantile(xf,.75))
    } else if (m$형태 == "log1p") cut_txt <- "log(1 + x)"
    covdict <- dplyr::bind_rows(covdict, data.frame(
      stratum = nm, code = v, variable_kr = m$kr, variable_en = m$eng, domain = m$cat,
      hypothesised_direction = m$이론방향, forced = m$forced,
      n = length(xf), mean = mean(xf), sd = sd(xf), min = min(xf),
      p25 = quantile(xf,.25), median = median(xf), p75 = quantile(xf,.75), max = max(xf),
      pct_zero = sum(xf == 0)/length(xf)*100,
      form_selected = m$형태, cutpoints = cut_txt,
      univariate_p = m$p, univariate_IRR = m$IRR,
      entered_model = as.integer(paste0(m$eng, "_z") %in% R$FMAP$safe),
      row.names = NULL))
  }
}

# ── 9. [N3] 형태 고정 + 스크리닝 없음 민감도 ★ 결론 유지 여부 판정 ────────────
cat("\n■ 9. [N3] 형태 고정·무스크리닝 민감도 ★\n")
fixed_form_fit <- function(DAT, label) {
  df <- DAT %>% filter(population > 0, region %in% shp_main$region)
  keep <- TV_PRESPEC$code[TV_PRESPEC$code %in% names(df)]
  vz <- c(); vmap <- data.frame()
  for (v in keep) {
    x <- suppressWarnings(as.numeric(df[[v]]))
    if (sum(is.finite(x)) < MIN_OBS || sd(x, na.rm = TRUE) == 0) next
    is_sex <- (v == "성비")
    form <- if (is_sex || is_pct(x)) "raw(z)" else "log1p(z)"
    xv <- if (form == "raw(z)") x else log1p(pmax(x, 0))
    s <- sd(xv, na.rm = TRUE); if (is.na(s) || s == 0) next
    eng <- TV_PRESPEC$eng[TV_PRESPEC$code == v][1]
    nmz <- paste0(eng, "_fz")
    df[[nmz]] <- (xv - mean(xv, na.rm = TRUE)) / s
    vz <- c(vz, nmz)
    vmap <- dplyr::bind_rows(vmap, data.frame(code = v, eng = eng, kr = TV_PRESPEC$kr[TV_PRESPEC$code == v][1],
                                              fixed_form = form, safe = nmz))
  }
  # VIF 정리만 적용 (p 스크리닝 없음)
  vd <- df[, c("cases", vz)]; vd <- vd[complete.cases(vd), ]
  fin <- vz
  for (stp in 1:40) {
    if (length(fin) <= 1) break
    lm_t <- tryCatch(lm(as.formula(paste("cases~", paste(paste0("`", fin, "`"), collapse = "+"))), data = vd),
                     error = function(e) NULL)
    if (is.null(lm_t)) break
    vv <- tryCatch(car::vif(lm_t), error = function(e) NULL); if (is.null(vv)) break
    names(vv) <- gsub("`", "", names(vv))
    if (max(vv, na.rm = TRUE) < VIF_THRESHOLD) break
    fin <- setdiff(fin, names(which.max(vv)))
  }
  rmap <- data.frame(region = shp_main$region, idarea = seq_len(nrow(shp_main)))
  ymap <- data.frame(year = YS:YE, idtime = seq_along(YS:YE))
  ic <- df[complete.cases(df[, fin]), ] %>%
    left_join(rmap, by = "region") %>% left_join(ymap, by = "year") %>% arrange(idarea, idtime)
  ic$idarea_time <- seq_len(nrow(ic))
  fs <- paste("cases ~", paste(fin, collapse = " + "), "+ offset(log(population+1))", NONSPATIAL)
  f <- fit_inla(fs, ic, fam = PRINCIPAL_FAMILY)
  if (is.null(f)) return(data.frame())
  t <- irr_tbl(f, label = "IRR_fixedform"); t$stratum <- label
  t$variable_kr <- vmap$kr[match(sub("_fz$", "", t$term), vmap$eng)]
  t$fixed_form  <- vmap$fixed_form[match(sub("_fz$", "", t$term), vmap$eng)]
  t$N <- nrow(ic); t$n_vars <- length(fin); t$DIC <- f$dic$dic
  cat(sprintf("  %s: 투입 %d변수, N=%d, DIC=%.1f, 유의 %d개\n",
              label, length(fin), nrow(ic), f$dic$dic, sum(t$sig == "*")))
  t
}
fixed_out <- dplyr::bind_rows(
  fixed_form_fit(DAT_all,   "Overall"),
  fixed_form_fit(DAT_urban, "Urban"),
  fixed_form_fit(DAT_rural, "Rural"))

# ── 10. [N6] PC prior 민감도 ─────────────────────────────────────────────────
cat("\n■ 10. [N6] PC prior 민감도 (u = 0.1 / 0.5 / 1.0)\n")
prior_sens <- data.frame()
for (u in c(0.1, 0.5, 1.0)) {
  pcp <- list(prec = list(prior = "pc.prec", param = c(u, 0.01)))
  ns <- sprintf("+ f(idtime,model='rw1',hyper=list(prec=list(prior='pc.prec',param=c(%g,0.01)))) + f(idarea,model='iid',hyper=list(prec=list(prior='pc.prec',param=c(%g,0.01))))", u, u)
  fs <- paste("cases ~", paste(res_all$FMAP$safe, collapse = " + "),
              "+ offset(log(population+1))", ns)
  f <- fit_inla(fs, res_all$ic, fam = PRINCIPAL_FAMILY)
  if (!is.null(f)) {
    t <- irr_tbl(f, label = "IRR"); t$pc_u <- u; t$DIC <- f$dic$dic
    t$variable_kr <- res_all$FMAP$kr[match(sub("_z$", "", t$term), res_all$FMAP$eng)]
    prior_sens <- dplyr::bind_rows(prior_sens, t)
    cat(sprintf("  u=%.1f: DIC=%.1f, 유의 %d개\n", u, f$dic$dic, sum(t$sig == "*")))
  }
}

# ── 11. 주모형 IRR 재출력(대조용) + 환경정보 ─────────────────────────────────
main_irr <- dplyr::bind_rows(lapply(names(STRATA), function(nm) {
  R <- STRATA[[nm]]; if (is.null(R$mv) || nrow(R$mv) == 0) return(data.frame())
  d <- R$mv; d$stratum <- nm; d }))

env_info <- data.frame(item = c("R", "INLA", "sf", "spdep", "date", "elapsed_min",
                                "n_districts_model", "n_obs_overall", "num.threads", "inla.mode"),
  value = c(R.version.string, as.character(packageVersion("INLA")),
            as.character(packageVersion("sf")), as.character(packageVersion("spdep")),
            format(Sys.time()), sprintf("%.1f", as.numeric(difftime(Sys.time(), T_START, units = "mins"))),
            nrow(shp_main), nrow(res_all$ic), "1:1", "compact"))

# ── 12. 저장 ─────────────────────────────────────────────────────────────────
cat("\n■ 12. 저장\n")
xlsx_path <- file.path(OUTDIR, sprintf("REV1_추가분석_%s.xlsx", TS))
wb <- openxlsx::createWorkbook()
add <- function(nm, df) { if (is.null(df) || nrow(df) == 0) return(invisible());
  openxlsx::addWorksheet(wb, nm); openxlsx::writeData(wb, nm, df) }
add("00_env",              env_info)
add("01_main_IRR_재현",    main_irr)
add("02_binary_part_OR",   bin_out)        # N1  → 보충표(이항부)
add("03_hurdle_IRR",       hurdle_out)     # N1b
add("04_IRR_unstandardised", unstd_out)    # N2
add("05_fixedform_sens",   fixed_out)      # N3 ★
add("06_covariate_dict",   covdict)        # N5
add("07_missing_audit",    audit)          # N4
add("08_hyper_posterior",  hyper_post)     # N7 (영과잉 확률 포함)
add("09_prior_spec_dump",  prior_spec)     # N6
add("10_prior_sensitivity", prior_sens)    # N6
add("11_zero_coding_diag", zero_diag)      # N0 ★ 결측-0 코딩 진단
add("12_corrected_IRR",    corrected_irr)  # N0 ★ 교정 패널 IRR
add("13_orig_vs_corrected", cmp_fix)       # N0 ★ 원본 vs 교정 대조
openxlsx::saveWorkbook(wb, xlsx_path, overwrite = TRUE)

saveRDS(list(res_all = res_all, res_urban = res_urban, res_rural = res_rural,
             fix_all = fix_all, fix_urban = fix_urban, fix_rural = fix_rural,
             bin_fit = bin_fit_list, audit = audit, covdict = covdict,
             zero_diag = zero_diag, cmp_fix = cmp_fix,
             fixed_out = fixed_out, prior_sens = prior_sens, env = env_info),
        file.path(OUTDIR, sprintf("REV1_objects_%s.rds", TS)))
writeLines(capture.output(sessionInfo()), file.path(OUTDIR, sprintf("sessionInfo_%s.txt", TS)))

cat(sprintf("\n✅ 완료: %s\n   소요 %.1f분\n", xlsx_path,
            as.numeric(difftime(Sys.time(), T_START, units = "mins"))))
cat("\n※ 확인 포인트 (중요도 순)\n")
cat("  1) 13_orig_vs_corrected ★★★ 결측-0 코딩 교정 후 유의성이 바뀐 변수가 있는가\n")
cat("  2) 05_fixedform_sens  ★★  형태 고정·무스크리닝에서도 주요 결과가 유지되는가\n")
cat("  3) 01_main_IRR_재현 의 N이 3,122인지 (원고 수치와 일치해야 함)\n")
cat("  4) 02_binary_part_OR 에서 logpop·의사수 등이 0 여부를 지배하는지 (과소보고 근거)\n")
cat("  5) 08_hyper_posterior 의 'zero-probability' 행 = 원고에 보고할 π 사후추정치\n")
cat("  6) 11_zero_coding_diag 에서 ZERO_INVALID 목록에 넣을 변수를 더 추가할지 판단\n")
