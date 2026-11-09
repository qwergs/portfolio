# ==============================================================================
#  REV1 : M2 기준 도시x농촌 상호작용 (Figure 4 / 보충표 교체용)
#  ------------------------------------------------------------------------
#  변수별로 (공변량 x urban) 모형을 따로 적합합니다.
#  성비·고령인구비율은 자기 자신이 아닌 한 보정으로 함께 투입 (원본 절차 동일).
#
#  실행:  Rscript -e 'source("REV1_M2_상호작용.R", encoding="UTF-8")'
#  (변수 수만큼 모형 적합 -> 5분 내외)
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

# fit_inla 는 헬퍼 3종에 없고 REV1_추가분석.R 에만 있으므로 여기서 동일하게 정의
fit_inla <- function(fs, dat, fam = PRINCIPAL_FAMILY, ...) {
  tryCatch(inla(as.formula(fs), family = fam, data = dat,
                control.compute = list(dic = TRUE, waic = TRUE, cpo = TRUE),
                control.predictor = list(link = 1), verbose = FALSE, ...),
           error = function(e) { cat(sprintf("\n    X fit 실패: %s\n", e$message)); NULL })
}


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






# ── 2. M2 패널·변수집합 ──────────────────────────────────────────────────────
cat("\n■ 2. M2 구성\n")
WATER <- "상수도보급률"; PHYS <- "의사수"; URBAN <- "도시지역인구비율"
PRE <- PRE_RAW
xu <- suppressWarnings(as.numeric(PRE[[URBAN]])); bad <- is.finite(xu) & xu > 100
cat(sprintf("  · %s 100 초과 → NA : %d 셀\n", URBAN, sum(bad)))
xu[bad] <- NA_real_; PRE[[URBAN]] <- xu
cm2 <- .orig_finalize(PRE, YS, YE)
cm2$area_3 <- sapply(cm2$region, classify_region)
cm2$area_2 <- ifelse(cm2$area_3 == "농촌", "농촌", "도시")
TV <- TV_PRESPEC[!TV_PRESPEC$code %in% c(WATER, PHYS), , drop = FALSE]; rownames(TV) <- NULL
cat(sprintf("  · 사전지정 %d개 (상수도·의사수 제외)\n", nrow(TV)))

# ── 3. 전체 모형 적합 → FMAP·ic 확보 ────────────────────────────────────────
cat("\n■ 3. 전체 모형 적합\n")
res <- run_model(TV, cm2, quiet = TRUE, re_str = NONSPATIAL, fam = PRINCIPAL_FAMILY)
stopifnot(!is.null(res$ic), !is.null(res$FMAP))
ic <- res$ic; FMAP <- res$FMAP; cov <- FMAP$safe
ic$urban <- as.integer(ic$area_2 == "도시")
cat(sprintf("  N=%d / 투입 변수 %d개 / DIC=%.1f\n", nrow(ic), length(cov), res$dic))

# ── 4. 변수별 도시×농촌 상호작용 (원본 analyze_period(full=TRUE) 절차 그대로) ─
cat("\n■ 4. 상호작용 (변수별 개별 모형)\n")
mk_kr2 <- function(cz, FM) { h <- FM$kr[FM$safe == cz]; if (length(h)) h[1] else cz }
forced <- intersect(c("sex_ratio_z","elderly_rate_z"), cov)
int_tbl <- data.frame()
for (cz in cov) {
  adj <- setdiff(forced, cz); rhs <- paste0(cz, " * urban")
  if (length(adj) > 0) rhs <- paste(rhs, "+", paste(adj, collapse = " + "))
  fs <- paste("cases ~", rhs, "+ offset(log(population+1))", NONSPATIAL)
  fit <- fit_inla(fs, ic, fam = PRINCIPAL_FAMILY)
  if (is.null(fit)) { cat(sprintf("\n    X %s 적합 실패\n", cz)); next }
  fe <- fit$summary.fixed
  irn <- grep(paste0("^", cz, ":urban$|^urban:", cz, "$"), rownames(fe), value = TRUE)
  if (!(cz %in% rownames(fe)) || length(irn) != 1) next
  br <- fe[cz, ]; bi <- fe[irn, ]
  int_tbl <- rbind(int_tbl, data.frame(
    term = cz, Variable = mk_kr2(cz, FMAP),
    IRR_rural = round(exp(br$mean), 4),
    rural_lo = round(exp(br$`0.025quant`), 4), rural_hi = round(exp(br$`0.975quant`), 4),
    IRR_urban_pt = round(exp(br$mean + bi$mean), 4),
    Interaction_IRR = round(exp(bi$mean), 4),
    inter_lo = round(exp(bi$`0.025quant`), 4), inter_hi = round(exp(bi$`0.975quant`), 4),
    Diff_sig = ifelse(bi$`0.025quant` > 0 | bi$`0.975quant` < 0, "★", ""),
    DIC = round(fit$dic$dic, 1), stringsAsFactors = FALSE))
  cat(".")
}
cat("\n")
print(int_tbl[, c("Variable","IRR_rural","IRR_urban_pt","Interaction_IRR","inter_lo","inter_hi","Diff_sig")],
      row.names = FALSE)
cat(sprintf("\n  ★ 도시·농촌 차이가 유의한 변수: %d개\n", sum(int_tbl$Diff_sig == "★")))
if (any(int_tbl$Diff_sig == "★"))
  cat(sprintf("     %s\n", paste(int_tbl$Variable[int_tbl$Diff_sig == "★"], collapse = ", ")))

# ── 5. 저장 ──────────────────────────────────────────────────────────────────
TS2 <- format(Sys.time(), "%y%m%d_%H%M")
xp <- file.path(OUTDIR, sprintf("REV1_M2_상호작용_%s.xlsx", TS2))
wb <- openxlsx::createWorkbook()
openxlsx::addWorksheet(wb, "34_interaction_M2"); openxlsx::writeData(wb, "34_interaction_M2", int_tbl)
openxlsx::addWorksheet(wb, "35_FMAP_M2");        openxlsx::writeData(wb, "35_FMAP_M2", FMAP)
openxlsx::saveWorkbook(wb, xp, overwrite = TRUE)
cat(sprintf("\n  v %s\n", xp))
cat("\n※ 투고본에서는 성비·민방위용지하수·인플루엔자·걷기실천율 4개가 유의했습니다.\n")
cat("   M2에서 몇 개가 유의한지, 그 집합이 같은지 확인하십시오.\n")
