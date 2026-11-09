# ==============================================================================
#  REV1 M2 공간구조 비교 — 비공간 vs ICAR vs BYM, ZINB/NB 양쪽
#  ------------------------------------------------------------------------
#  변경 1. 상수도보급률을 사전지정 변수에서 제외
#     · 2018-2024 매년 76개 지역이 0 (결측의 0-코딩, 오염 구간이 절반)
#     · 저자 출처표에 1차 출처가 문서화되어 있지 않음 (R2-6/7/8 답변 불가)
#     · 투고본에서 비유의(log1p 1.096, 0.79-1.53) → 결론 영향 없음 예상
#  변경 2. 의사수를 사전지정 변수에서 제외
#     · 2017년 통합시 12곳이 일제히 -46~-50% (2017/2016 중앙값 0.517)
#       나머지 217곳은 0.7 미만이 0곳. 전국 합계 115,112 -> 100,241 (-15,000명)
#     · 2024년은 12곳 0 + 16곳 결측
#     · 출처표에 'KOSIS 시도별 의료인력 현황'(광역 단위)으로 기재
#     · 투고본·M 모두 비유의 -> 결론 영향 없음 예상 (본 실행으로 검증)
#  변경 3. 도시지역인구비율 100 초과 10셀을 NA 처리
#     · 전부 2020년 · 전부 자치구를 둔 통합시
#     · 분자는 시 단위 · 분모는 구 단위로 집계되어 자치구 수만큼 부풀음
#     · 전후 연도가 평탄하므로 기존 보간 규칙이 참값에 가깝게 메움
#
#  실행:  Rscript -e 'source("REV1_M2_공간비교.R", encoding="UTF-8")'
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

# ── 0b. ZENODO 기탁본에 없는 보조함수 (REV1_추가분석.R 정의를 그대로 복사) ──────
#   irr_tbl 은 REV1_추가분석.R 에만 정의되어 있고 기탁본 3개 소스에는 없습니다.
#   이 스크립트는 기탁본만 source 하므로 여기에 동일 정의를 둡니다.
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
cat("  · 보조함수 irr_tbl 정의 완료\n")

# ── 0c. PIT 도우미 (REV1_PIT_보정진단.R 정의를 그대로 복사) ────────────────────
.pit_parts <- function(fit) {
  ok  <- which(is.finite(fit$cpo$pit) & is.finite(fit$cpo$cpo) &
               (is.null(fit$cpo$failure) | fit$cpo$failure == 0))
  pit <- pmin(1, pmax(0, fit$cpo$pit[ok])); cpo <- pmin(1, pmax(0, fit$cpo$cpo[ok]))
  lo  <- pmax(0, pit - cpo); hi <- pmax(lo, pit)
  list(pit = pit, cpo = cpo, lo = lo, hi = hi, n = length(ok))
}
pit_hist <- function(P, J = 20) {
  u <- seq(0, 1, length.out = J + 1)
  Fb <- vapply(u, function(uu)
    mean(pmin(1, pmax(0, (uu - P$lo) / pmax(P$hi - P$lo, 1e-12)))), numeric(1))
  J * diff(Fb)
}
rand_pit <- function(P, seed = 20260915) { set.seed(seed); P$lo + runif(P$n) * (P$hi - P$lo) }
pit_row <- function(fit, stratum, fam) {
  if (is.null(fit) || is.null(fit$cpo)) return(data.frame())
  P <- .pit_parts(fit); mid <- P$pit - 0.5 * P$cpo; rp <- rand_pit(P); h <- pit_hist(P, 20)
  data.frame(stratum = stratum, likelihood = fam, n = P$n,
             DIC = round(fit$dic$dic, 2), WAIC = round(fit$waic$waic, 2),
             mean_PIT_nonrandomised = round(mean(P$pit), 4),
             mean_CPO               = round(mean(P$cpo), 4),
             mean_PIT_mid           = round(mean(mid), 4),
             mean_PIT_randomised    = round(mean(rp), 4),
             hist_max_dev_from_1    = round(max(abs(h - 1)), 3),
             hist_rmse_from_1       = round(sqrt(mean((h - 1)^2)), 3),
             ks_p_randomised = tryCatch(round(suppressWarnings(
                 ks.test(rp, "punif")$p.value), 4), error = function(e) NA))
}
cat("  · PIT 도우미 정의 완료\n")

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





# ── 2. 패널 2종 · 변수집합 2종 ────────────────────────────────────────────────
cat("\n■ 2. 패널·변수집합 구성\n")
WATER  <- "상수도보급률"
URBAN  <- "도시지역인구비율"
PHYS   <- "의사수"

# P0 = 투고본 그대로
PRE_P0 <- PRE_RAW

# P1 = P0 + 도시지역인구비율 100 초과(2020년 통합시 10셀) NA 처리
PRE_P1 <- PRE_P0
xu <- suppressWarnings(as.numeric(PRE_P1[[URBAN]]))
bad <- is.finite(xu) & xu > 100
cat(sprintf("  · %s 100 초과 → NA : %d 셀 (연도: %s)\n", URBAN, sum(bad),
            paste(sort(unique(PRE_P1$year[bad])), collapse = ", ")))
xu[bad] <- NA_real_; PRE_P1[[URBAN]] <- xu

mk <- function(P) { z <- .orig_finalize(P, YS, YE)
  z$area_3 <- sapply(z$region, classify_region)
  z$area_2 <- ifelse(z$area_3 == "농촌", "농촌", "도시"); z }
cm_P0 <- mk(PRE_P0); cm_P1 <- mk(PRE_P1)

# ZERO_INVALID 교정판 (N0 민감도)
ZERO_INVALID <- c("상수도보급률","하수도보급률","의사수","관내진료비_전체",
                  "연간인플루엔자예방접종률_표준화율","건강생활실천율_조율",
                  "걷기실천율_표준화율","재정자립도","재정자주도","성비","고령인구비율")
PRE_P2 <- PRE_P1
nz <- 0
for (v in intersect(ZERO_INVALID, names(PRE_P2))) {
  x <- suppressWarnings(as.numeric(PRE_P2[[v]])); nz <- nz + sum(is.finite(x) & x == 0)
  x[is.finite(x) & x == 0] <- NA_real_; PRE_P2[[v]] <- x
}
cat(sprintf("  · ZERO_INVALID 0 → NA : %d 셀\n", nz))
cm_P2 <- mk(PRE_P2)

TV_A <- TV_PRESPEC                                                       # 21개 (투고본)
TV_B <- TV_PRESPEC[!TV_PRESPEC$code %in% c(WATER), , drop = FALSE]       # 20개 (상수도 제외)
TV_C <- TV_PRESPEC[!TV_PRESPEC$code %in% c(WATER, PHYS), , drop = FALSE] # 19개 (+ 의사수 제외)
rownames(TV_B) <- NULL; rownames(TV_C) <- NULL
cat(sprintf("  · 변수집합 A(투고본) %d / B(%s 제외) %d / C(+%s 제외) %d\n",
            nrow(TV_A), WATER, nrow(TV_B), PHYS, nrow(TV_C)))

# ── 3. 공간구조 3종 x 우도 2종, M2(19변수) Overall ───────────────────────────
cat("\n■ 3. 공간구조 비교 (M2, 19변수, Overall)\n")
pc_bym <- list(prec.unstruct = list(prior = "pc.prec", param = c(0.5, 0.01)),
               prec.spatial  = list(prior = "pc.prec", param = c(0.5, 0.01)))
RE <- list(
  NonSpatial = NONSPATIAL,
  ICAR = "+ f(idtime,model='rw1',hyper=pc_prec) + f(idarea,model='besag',graph=g_main,scale.model=TRUE,hyper=pc_prec)",
  BYM  = "+ f(idtime,model='rw1',hyper=pc_prec) + f(idarea,model='bym',graph=g_main,scale.model=TRUE,hyper=pc_bym)")
FAMS <- c(ZINB = "zeroinflatednbinomial1", NB = "nbinomial")

out <- data.frame()
for (rk in names(RE)) for (fk in names(FAMS)) {
  r <- tryCatch(run_model(TV_C, cm_P1, quiet = TRUE, re_str = RE[[rk]], fam = FAMS[[fk]]),
                error = function(e) { cat("   ! ", rk, fk, conditionMessage(e), "\n"); NULL })
  if (is.null(r) || is.null(r$fit)) { cat(sprintf("  %-11s %-5s 적합 실패\n", rk, fk)); next }
  f <- r$fit
  out <- rbind(out, data.frame(structure = rk, likelihood = fk,
    n = nrow(r$ic), n_vars = nrow(r$FMAP),
    DIC = round(f$dic$dic, 2), WAIC = round(f$waic$waic, 2)))
  cat(sprintf("  %-11s %-5s DIC=%.2f WAIC=%.2f\n", rk, fk, f$dic$dic, f$waic$waic))
}

cat("\n■ 4. 요약\n")
if (nrow(out)) {
  for (fk in c("ZINB","NB")) {
    s <- out[out$likelihood == fk, ]
    if (!nrow(s)) next
    base_dic <- s$DIC[s$structure == "NonSpatial"]
    s$dDIC_vs_nonspatial <- round(s$DIC - base_dic, 2)
    print(s, row.names = FALSE); cat("\n")
  }
  cat("  ※ 원고 결과절의 'spatial-model DIC 1,533.5-1,537.3' 은 투고본 21변수 + NB 계열 값입니다.\n")
  cat("     위 표의 ZINB 행으로 교체하거나, 비교가 NB 계열이었음을 문장에 명시하십시오.\n")
}

cat("\n■ 5. 저장\n")
TS5 <- format(Sys.time(), "%y%m%d_%H%M")
xp <- file.path(OUTDIR, sprintf("REV1_M2_공간비교_%s.xlsx", TS5))
wb <- openxlsx::createWorkbook()
openxlsx::addWorksheet(wb, "60_spatial_compare_M2"); openxlsx::writeData(wb, "60_spatial_compare_M2", out)
openxlsx::saveWorkbook(wb, xp, overwrite = TRUE)
cat(sprintf("  ✔ %s\n", xp))
cat("\n※ 확인\n")
cat("  1) ZINB 계열에서 ICAR/BYM 의 DIC 가 NonSpatial 보다 낮지 않아야 함 (공간항 기각 근거)\n")
cat("  2) ZINB NonSpatial DIC 가 1482.9 로 재현되는지\n")
