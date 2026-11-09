# ==============================================================================
#  REV1 M2 이항부(zero part) 재적합 — R2-14 대응
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
#  실행:  Rscript -e 'source("REV1_M2_이항부.R", encoding="UTF-8")'
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

# ── 3. 이항부 (any case vs zero) — M2 19변수 기준 ─────────────────────────────
cat("\n■ 3. 이항부 로지스틱 (any case vs zero) — M2 (19변수)\n")

# run_model 과 동일한 전처리를 거친 ic / FMAP 을 얻기 위해 주분석을 한 번 적합
prep <- list(
  Overall = run_model(TV_C, cm_P1,                                    quiet = TRUE, re_str = NONSPATIAL, fam = PRINCIPAL_FAMILY),
  Urban   = run_model(TV_C, cm_P1 %>% dplyr::filter(area_2 == "도시"), quiet = TRUE, re_str = NONSPATIAL, fam = PRINCIPAL_FAMILY),
  Rural   = run_model(TV_C, cm_P1 %>% dplyr::filter(area_2 == "농촌"), quiet = TRUE, re_str = NONSPATIAL, fam = PRINCIPAL_FAMILY))

bin_out <- data.frame(); hur_out <- data.frame()
for (nm in names(prep)) {
  R <- prep[[nm]]; if (is.null(R$ic)) next
  ic <- R$ic
  ic$any    <- as.integer(ic$cases > 0)
  ic$logpop <- as.numeric(scale(log(ic$population + 1)))
  cov_str <- paste(R$FMAP$safe, collapse = " + ")
  fs <- paste("any ~", cov_str, "+ logpop", NONSPATIAL)
  fb <- tryCatch(inla(as.formula(fs), family = "binomial", Ntrials = 1, data = ic,
                      control.compute = list(dic = TRUE, waic = TRUE, cpo = TRUE),
                      control.predictor = list(link = 1)), error = function(e) NULL)
  if (!is.null(fb)) {
    t <- irr_tbl(fb, label = "OR"); t$stratum <- nm
    t$variable_kr <- vapply(t$term, function(z) {
      rn <- sub("_z$", "", z); k <- R$FMAP$kr[R$FMAP$eng == rn]
      if (length(k)) k[1] else rn }, character(1))
    t$n_obs <- nrow(ic); t$zero_rate <- mean(ic$any == 0); t$DIC <- fb$dic$dic
    bin_out <- dplyr::bind_rows(bin_out, t)
    cat(sprintf("  %-8s N=%d, 0건 비율=%.3f, DIC=%.1f, 유의 %d개\n",
                nm, nrow(ic), mean(ic$any == 0), fb$dic$dic, sum(t$sig == "*")))
  } else cat(sprintf("  %-8s 적합 실패\n", nm))

  # 허들(zeroinflatednbinomial0) 병행 — 0 초과 부분의 계수부
  fh <- tryCatch(run_model(TV_C, if (nm == "Overall") cm_P1 else
                             cm_P1 %>% dplyr::filter(area_2 == ifelse(nm == "Urban", "도시", "농촌")),
                           quiet = TRUE, re_str = NONSPATIAL, fam = "zeroinflatednbinomial0"),
                 error = function(e) NULL)
  if (!is.null(fh) && !is.null(fh$mv) && nrow(fh$mv)) {
    d <- fh$mv; d$stratum <- nm; d$DIC <- fh$dic
    hur_out <- dplyr::bind_rows(hur_out, d)
  }
}

# ── 4. 계수부 IRR 과 나란히 보기 ──────────────────────────────────────────────
cat("\n■ 4. 계수부(IRR) vs 이항부(OR) 대조\n")
cnt <- dplyr::bind_rows(lapply(names(prep), function(s) {
  d <- prep[[s]]$mv; if (is.null(d) || !nrow(d)) return(data.frame())
  d$stratum <- s; d }))
side <- data.frame()
if (nrow(cnt) && nrow(bin_out)) {
  b <- bin_out; b$var_kr <- b$variable_kr
  side <- merge(cnt[, c("stratum","var_kr","형태","IRR","lo","hi","sig")],
                b[,   c("stratum","var_kr","OR","OR_lo","OR_hi","sig")],
                by = c("stratum","var_kr"), all = TRUE, suffixes = c("_count","_zero"))
  side$방향상반 <- with(side, ifelse(is.na(IRR) | is.na(OR), "",
                       ifelse((IRR > 1 & OR < 1) | (IRR < 1 & OR > 1), "OPPOSITE", "")))
  keep <- (!is.na(side$sig_count) & side$sig_count == "*") |
          (!is.na(side$sig_zero)  & side$sig_zero  == "*")
  if (any(keep)) print(side[keep, c("stratum","var_kr","IRR","sig_count","OR","sig_zero","방향상반")],
                       row.names = FALSE) else cat("  (유의 항목 없음)\n")
}

# ── 5. 저장 ──────────────────────────────────────────────────────────────────
cat("\n■ 5. 저장\n")
TS3 <- format(Sys.time(), "%y%m%d_%H%M")
xp <- file.path(OUTDIR, sprintf("REV1_M2_이항부_%s.xlsx", TS3))
wb <- openxlsx::createWorkbook()
add <- function(nm, df) { if (is.null(df) || !nrow(df)) return(invisible())
  openxlsx::addWorksheet(wb, nm); openxlsx::writeData(wb, nm, df) }
add("40_binary_part_OR_M2", bin_out)
add("41_hurdle_count_M2",   hur_out)
add("42_count_vs_zero",     side)
openxlsx::saveWorkbook(wb, xp, overwrite = TRUE)
cat(sprintf("  ✔ %s\n", xp))
cat("\n※ 확인 순서\n")
cat("  1) 40_binary_part_OR_M2 — 0건 여부를 지배하는 변수 (R2-14 직접 답변)\n")
cat("  2) 42_count_vs_zero 의 '방향상반' 열 — 걷기실천율이 OPPOSITE 여야 함\n")
cat("  3) 41_hurdle_count_M2 — 허들 모형의 계수부가 주분석과 일치하는지\n")
