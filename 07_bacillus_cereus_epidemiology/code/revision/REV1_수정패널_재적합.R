# ==============================================================================
#  REV1 수정 패널 재적합
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
#  실행:  Rscript -e 'source("REV1_수정패널_재적합.R", encoding="UTF-8")'
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

# ── 3. 적합 ──────────────────────────────────────────────────────────────────
cat("\n■ 3. 적합 (12건)\n")
fitset <- function(TV, PANEL) list(
  Overall = run_model(TV, PANEL,                                     quiet = TRUE, re_str = NONSPATIAL, fam = PRINCIPAL_FAMILY),
  Urban   = run_model(TV, PANEL %>% dplyr::filter(area_2 == "도시"), quiet = TRUE, re_str = NONSPATIAL, fam = PRINCIPAL_FAMILY),
  Rural   = run_model(TV, PANEL %>% dplyr::filter(area_2 == "농촌"), quiet = TRUE, re_str = NONSPATIAL, fam = PRINCIPAL_FAMILY))

CFG <- list(
  "R_투고본재현"       = list(TV = TV_A, P = cm_P0),
  "M_상수도제외"       = list(TV = TV_B, P = cm_P1),
  "M2_상수도+의사수제외" = list(TV = TV_C, P = cm_P1),
  "S2_M2+영코딩교정"   = list(TV = TV_C, P = cm_P2))
FITS <- list()
for (nm in names(CFG)) { FITS[[nm]] <- fitset(CFG[[nm]]$TV, CFG[[nm]]$P); cat(sprintf("  · %s 완료\n", nm)) }

# ── 4. 진단표 ────────────────────────────────────────────────────────────────
cat("\n■ 4. N / DIC / 유의\n")
dg <- data.frame()
for (nm in names(FITS)) for (s in c("Overall","Urban","Rural")) {
  f <- FITS[[nm]][[s]]
  fe <- if (is.null(f$fit)) NULL else {
    z <- f$fit$summary.fixed; z[rownames(z) != "(Intercept)", , drop = FALSE] }
  dg <- rbind(dg, data.frame(config = nm, stratum = s,
    N = ifelse(is.null(f$ic), NA, nrow(f$ic)),
    n_vars = ifelse(is.null(f$FMAP), NA, nrow(f$FMAP)),
    DIC = ifelse(is.null(f$dic), NA, round(f$dic, 1)),
    n_sig = ifelse(is.null(f$sig_count), NA, f$sig_count),
    max_abs_logIRR = ifelse(is.null(fe), NA, round(max(abs(fe$mean)), 3)),
    판정 = ifelse(is.null(fe), "FIT 실패",
            ifelse(max(abs(fe$mean)) < 0.20, "★계수 전면 수축 = 수렴 의심", "정상"))))
}
print(dg, row.names = FALSE)

bind <- function(F, tag) dplyr::bind_rows(lapply(names(F), function(s) {
  d <- F[[s]]$mv; if (is.null(d) || !nrow(d)) return(data.frame())
  d$stratum <- s; d$config <- tag; d }))
irr <- dplyr::bind_rows(lapply(names(FITS), function(nm) bind(FITS[[nm]], nm)))

cmp <- data.frame()
for (s in c("Overall","Urban","Rural")) {
  a <- FITS[["R_투고본재현"]][[s]]$mv; b <- FITS[["M2_상수도+의사수제외"]][[s]]$mv
  if (is.null(a) || is.null(b) || !nrow(a) || !nrow(b)) next
  m <- merge(a[, c("var_kr","형태","IRR","lo","hi","sig")],
             b[, c("var_kr","형태","IRR","lo","hi","sig")],
             by = "var_kr", all = TRUE, suffixes = c("_R","_M2"))
  m$stratum <- s
  m$flip <- with(m, ifelse(is.na(sig_R) | is.na(sig_M2), "only_one_model",
                    ifelse(sig_R == sig_M2, "", "SIGNIFICANCE_CHANGED")))
  cmp <- dplyr::bind_rows(cmp, m)
}
chg <- if (nrow(cmp)) cmp[cmp$flip != "", , drop = FALSE] else cmp
cat(sprintf("\n  ★ 투고본(R) → 최종 주분석(M2)에서 유의성이 바뀐 항목: %d건\n", nrow(chg)))
if (nrow(chg)) print(chg[, c("stratum","var_kr","IRR_R","sig_R","IRR_M2","sig_M2","flip")], row.names = FALSE)

# ── 4b. M vs M2 : 의사수 제외의 순수 효과 ────────────────────────────────────
cmp2 <- data.frame()
for (s in c("Overall","Urban","Rural")) {
  a <- FITS[["M_상수도제외"]][[s]]$mv; b <- FITS[["M2_상수도+의사수제외"]][[s]]$mv
  if (is.null(a) || is.null(b) || !nrow(a) || !nrow(b)) next
  m <- merge(a[, c("var_kr","형태","IRR","lo","hi","sig")],
             b[, c("var_kr","형태","IRR","lo","hi","sig")],
             by = "var_kr", all = TRUE, suffixes = c("_M","_M2"))
  m$stratum <- s
  m$flip <- with(m, ifelse(is.na(sig_M) | is.na(sig_M2), "only_one_model",
                    ifelse(sig_M == sig_M2, "", "SIGNIFICANCE_CHANGED")))
  cmp2 <- dplyr::bind_rows(cmp2, m)
}
chg2 <- if (nrow(cmp2)) cmp2[cmp2$flip == "SIGNIFICANCE_CHANGED", , drop = FALSE] else cmp2
cat(sprintf("\n  ★ 의사수 제외(M → M2)로 유의성이 바뀐 항목: %d건\n", nrow(chg2)))
if (nrow(chg2)) print(chg2[, c("stratum","var_kr","IRR_M","sig_M","IRR_M2","sig_M2")], row.names = FALSE) else
  cat("     → 없음. 의사수 제외가 결론에 영향을 주지 않습니다.\n")

# ── 5. 저장 ──────────────────────────────────────────────────────────────────
cat("\n■ 5. 저장\n")
TS2 <- format(Sys.time(), "%y%m%d_%H%M")
xp <- file.path(OUTDIR, sprintf("REV1_수정패널_%s.xlsx", TS2))
wb <- openxlsx::createWorkbook()
add <- function(nm, df) { if (is.null(df) || !nrow(df)) return(invisible())
  openxlsx::addWorksheet(wb, nm); openxlsx::writeData(wb, nm, df) }
add("30_fit_진단", dg)
add("31_IRR_전체", irr)
add("32_R_vs_M2", cmp)
add("33_M_vs_M2", cmp2)
openxlsx::saveWorkbook(wb, xp, overwrite = TRUE)
saveRDS(list(FITS = FITS, dg = dg, cmp = cmp, cmp2 = cmp2, irr = irr),
        file.path(OUTDIR, sprintf("REV1_수정패널_objects_%s.rds", TS2)))
cat(sprintf("  ✔ %s\n", xp))
cat("\n※ 확인 순서\n")
cat("  1) 30_fit_진단 의 '판정' 열 — 수렴 실패가 없어야 함\n")
cat("  2) R / M / M2 의 Overall N 이 모두 3,122 여야 정상\n")
cat("  3) ★ M → M2 변화 0건이면 의사수 제외가 결론에 영향 없음 (제거 근거)\n")
cat("  4) ★ R → M2 변화가 최종 보고 대상\n")
