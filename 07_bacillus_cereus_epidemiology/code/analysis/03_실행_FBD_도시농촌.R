# ==============================================================================
# 03_실행_FBD_도시농촌.R — 전 질병 도시/농촌 공간분석 실행 러너
#
#   실행: source("03_실행_FBD_도시농촌.R")  (절대경로 source라 작업폴더 무관)
#   흐름: 질병 루프 → ① 사전 Moran 분류(O/X) → ② 3그룹 사전지정 INLA(AUTO 없음)
#         → 질병별 결과 xlsx 저장 → 마스터 비교표(공간상관 O/X × 도시/농촌 유의성)
#   베이스 2011–2024(TV_PRESPEC) / 식품 서브 2020–2024(TV_PRESPEC_SUB, RUN_SENSITIVITY)
# ==============================================================================
rm(list = ls()); gc()
CODE_DIR <- "/Users/guns/Desktop/도시_농촌_분석/code"
source(file.path(CODE_DIR, "00_config_공통.R"))
source(file.path(CODE_DIR, "01_데이터로드_함수.R"))
source(file.path(CODE_DIR, "02_모델_함수.R"))

RUN_SENSITIVITY <- TRUE   # 2020–2024 식품변수 민감도 실행 여부
TS <- format(Sys.time(), "%y%m%d_%H%M")

# ── 공간객체 (질병 무관, 1회 생성) ──
shp <- st_read(PATH_SHP, quiet = TRUE) %>%
  mutate(region = str_replace_all(as.character(region), "\\s+", ""),
         region = if_else(region == "인천시미추홀구", "인천시남구", region))
shp_main <- shp %>% filter(!region %in% c("인천시옹진군","전라남도완도군","전라남도진도군",
                                          "경상남도거제시","경상남도남해군","경상북도울릉군"))
nb_obj <- poly2nb(shp_main, snap = 0.01); iso <- which(card(nb_obj) == 0)
if (length(iso) > 0) { shp_main <- shp_main[-iso, ]; nb_obj <- poly2nb(shp_main, snap = 0.01) }
nb2INLA(nb_obj, file = "/tmp/fbd.graph"); g_main <- inla.read.graph("/tmp/fbd.graph")
nb_w <- nb2listw(nb_obj, style = "W")
cat(sprintf("시군구: %d개\n\n", nrow(shp_main)))

# ── 공통 base TV (강제 보정변수만; 나머지는 AUTO) ──
make_base_tv <- function() data.frame(
  tier = "A", cat = sapply(FORCED_VARS, assign_category), kr = FORCED_VARS, code = FORCED_VARS,
  eng = c("성비"="sex_ratio","고령인구비율"="elderly_rate")[FORCED_VARS],
  forced = "‡", 이론방향 = sapply(FORCED_VARS, assign_theory_dir), stringsAsFactors = FALSE)

# ── 한 질병 분석 ──
analyze_disease <- function(dz, ys, ye, include_food, tag) {
  CUR_YS <<- ys; CUR_YE <<- ye
  cat(sprintf("\n══════ %s [%s] %d–%d ══════\n", dz, tag, ys, ye))
  cm <- build_cor_merged(dz, ys, ye, include_food = include_food)
  # ① 사전 Moran
  mor <- moran_prescreen(cm)
  cat(sprintf("  사전 Moran I=%s p=%s → %s\n",
      ifelse(is.na(mor$I),"NA",sprintf("%+.4f",mor$I)),
      ifelse(is.na(mor$p),"NA",sprintf("%.4f",mor$p)), mor$verdict))
  # ② 3그룹 사전지정(PRE-SPECIFIED, AUTO 제거) — 고정 이론셋 1회 적합
  #    베이스(2011–2024)=TV_PRESPEC, 서브(2020–2024,식품)=TV_PRESPEC_SUB
  DAT_all <- cm; DAT_urban <- cm %>% filter(area_2 == "도시"); DAT_rural <- cm %>% filter(area_2 == "농촌")
  tv <- if (include_food) TV_PRESPEC_SUB else TV_PRESPEC
  res_all   <- run_model(tv, DAT_all,   quiet = TRUE)
  res_urban <- run_model(tv, DAT_urban, quiet = TRUE)
  res_rural <- run_model(tv, DAT_rural, quiet = TRUE)
  # 사후 Moran (전체 모델 잔차)
  post_I <- post_p <- NA
  if (!is.null(res_all$fit) && !is.null(res_all$ic)) {
    icf <- res_all$ic; resid <- icf$cases - res_all$fit$summary.fitted.values$mean[1:nrow(icf)]
    rdf <- data.frame(region = icf$region, r = resid) %>% group_by(region) %>% summarise(r = mean(r), .groups = "drop")
    rv <- rdf$r[match(shp_main$region, rdf$region)]; rv[is.na(rv)] <- 0
    mp <- tryCatch(moran.test(rv, nb_w), error = function(e) NULL)
    if (!is.null(mp)) { post_I <- unname(mp$estimate[1]); post_p <- mp$p.value }
  }
  list(disease = dz, tag = tag, ys = ys, ye = ye, moran = mor, post_I = post_I, post_p = post_p,
       res_all = res_all, res_urban = res_urban, res_rural = res_rural, n_vars = nrow(tv))
}

# ── 결과 저장 (질병별 xlsx: 3그룹 mv 표) ──
save_disease_xlsx <- function(R) {
  slug <- disease_slug(R$disease); en <- DISEASE_EN[R$disease]; if (is.na(en)) en <- R$disease
  outdir <- file.path(DIR_RES, slug); if (!dir.exists(outdir)) dir.create(outdir, recursive = TRUE)
  wb <- createWorkbook()
  s_t <- createStyle(fontSize = 13, textDecoration = "bold")
  s_h <- createStyle(fontSize = 10, textDecoration = "bold", halign = "center", border = "TopBottom", fgFill = "#D9E2F3")
  s_s <- createStyle(textDecoration = "bold", fontColour = "#C00000")
  write_grp <- function(ws, grp_en, res) {
    addWorksheet(wb, ws)
    writeData(wb, ws, sprintf("Table. Bayesian spatiotemporal model (M6 BYM+RW1+IID, NB) — %s, %s [%d-%d]", en, grp_en, R$ys, R$ye), startRow = 1)
    addStyle(wb, ws, s_t, rows = 1, cols = 1)
    if (is.null(res$mv) || nrow(res$mv) == 0) { writeData(wb, ws, "No results", startRow = 3); return() }
    hd <- c("Category","Variable","Form","IRR","95% CrI Lo","95% CrI Hi","Sig","Observed","Theory","Match")
    writeData(wb, ws, t(hd), startRow = 3, startCol = 1, colNames = FALSE); addStyle(wb, ws, s_h, rows = 3, cols = 1:10, gridExpand = TRUE)
    mv <- res$mv
    for (i in 1:nrow(mv)) { r <- 3 + i
      vals <- c(mv$카테고리[i], mv$var_kr[i], mv$형태[i], mv$IRR[i], mv$lo[i], mv$hi[i], mv$sig[i], mv$obs_dir[i], mv$theory_dir[i], mv$방향일치[i])
      for (cc in 1:10) writeData(wb, ws, vals[cc], startRow = r, startCol = cc)
      if (mv$sig[i] == "★") addStyle(wb, ws, s_s, rows = r, cols = 4:7, gridExpand = TRUE) }
    sr <- 4 + nrow(mv) + 1
    writeData(wb, ws, sprintf("Sig %d/%d | Forward %d | Reverse %d | DIC %s",
      ifelse(is.null(res$sig_count),0,res$sig_count), nrow(mv),
      ifelse(is.null(res$n_fwd),0,res$n_fwd), ifelse(is.null(res$n_rev),0,res$n_rev),
      ifelse(is.null(res$dic),"NA",sprintf("%.2f",res$dic))), startRow = sr, startCol = 1)
  }
  write_grp("Overall", "Overall", R$res_all)
  write_grp("Urban",   "Urban",   R$res_urban)
  write_grp("Rural",   "Rural",   R$res_rural)
  fn <- file.path(outdir, sprintf("%s_도시농촌_%s_%s.xlsx", slug, R$tag, TS))
  saveWorkbook(wb, fn, overwrite = TRUE)
  cat(sprintf("  ★ 저장: %s\n", fn))
}

# ══════════════════════════════════════════
# 메인 루프
# ══════════════════════════════════════════
master <- data.frame()
for (dz in DISEASES_MAIN) {
  R <- tryCatch(analyze_disease(dz, MAIN_START, MAIN_END, include_food = FALSE, tag = "main"),
                error = function(e) { cat(sprintf("  ❌ %s: %s\n", dz, e$message)); NULL })
  if (is.null(R)) next
  save_disease_xlsx(R)
  g <- function(x, f) ifelse(is.null(x[[f]]), NA, x[[f]])
  master <- rbind(master, data.frame(
    disease = DISEASE_EN[dz], disease_kr = dz,
    moran_I = R$moran$I, moran_p = R$moran$p, spatial = ifelse(is.na(R$moran$p), NA, R$moran$p < 0.05),
    post_moran_I = R$post_I, post_moran_p = R$post_p,
    n_vars = R$n_vars,
    all_sig = g(R$res_all,"sig_count"),   all_fwd = g(R$res_all,"n_fwd"),
    urban_sig = g(R$res_urban,"sig_count"), urban_fwd = g(R$res_urban,"n_fwd"),
    rural_sig = g(R$res_rural,"sig_count"), rural_fwd = g(R$res_rural,"n_fwd"),
    all_dic = g(R$res_all,"dic"),
    stringsAsFactors = FALSE))
  gc()
}

# ── 마스터 비교표 저장 (Step-3: 공간상관 O/X × 도시/농촌 유의성) ──
wb <- createWorkbook()
s_t <- createStyle(fontSize = 13, textDecoration = "bold")
s_h <- createStyle(fontSize = 10, textDecoration = "bold", halign = "center", fgFill = "#D9E2F3")
addWorksheet(wb, "Master_Comparison")
writeData(wb, "Master_Comparison", sprintf("FBD urban/rural spatial analysis — master comparison (main %d-%d)", MAIN_START, MAIN_END), startRow = 1)
addStyle(wb, "Master_Comparison", s_t, rows = 1, cols = 1)
master_disp <- master %>% arrange(desc(spatial), moran_p)
writeData(wb, "Master_Comparison", master_disp, startRow = 3, colNames = TRUE)
addStyle(wb, "Master_Comparison", s_h, rows = 3, cols = 1:ncol(master_disp), gridExpand = TRUE)
# 그룹 요약 (공간상관 O vs X 평균 유의변수 수)
addWorksheet(wb, "GroupA_vs_B")
ga <- master %>% filter(spatial == TRUE); gb <- master %>% filter(spatial == FALSE)
summ <- data.frame(
  Group = c("A (spatial, p<0.05)", "B (non-spatial)"),
  N = c(nrow(ga), nrow(gb)),
  mean_all_sig = c(mean(ga$all_sig, na.rm=TRUE), mean(gb$all_sig, na.rm=TRUE)),
  mean_urban_sig = c(mean(ga$urban_sig, na.rm=TRUE), mean(gb$urban_sig, na.rm=TRUE)),
  mean_rural_sig = c(mean(ga$rural_sig, na.rm=TRUE), mean(gb$rural_sig, na.rm=TRUE)))
writeData(wb, "GroupA_vs_B", "Spatial (A) vs Non-spatial (B): mean significant variables by group", startRow = 1)
addStyle(wb, "GroupA_vs_B", s_t, rows = 1, cols = 1)
writeData(wb, "GroupA_vs_B", summ, startRow = 3, colNames = TRUE)
addStyle(wb, "GroupA_vs_B", s_h, rows = 3, cols = 1:ncol(summ), gridExpand = TRUE)
fn_master <- file.path(DIR_RES, sprintf("00_MASTER_요약_%s.xlsx", TS))
saveWorkbook(wb, fn_master, overwrite = TRUE)
cat(sprintf("\n★ 마스터 비교표: %s\n", fn_master))

# ══════════════════════════════════════════
# 민감도 (2020–2024 + 식품변수) — 옵션
# ══════════════════════════════════════════
if (RUN_SENSITIVITY) {
  cat("\n\n##### 민감도 분석 (2020–2024 + 식품변수) #####\n")
  for (dz in DISEASES_MAIN) {
    R <- tryCatch(analyze_disease(dz, SENS_START, SENS_END, include_food = TRUE, tag = "sens2020"),
                  error = function(e) { cat(sprintf("  ❌ %s: %s\n", dz, e$message)); NULL })
    if (!is.null(R)) save_disease_xlsx(R)
    gc()
  }
}

cat("\n═══ FBD 도시/농촌 통합 분석 완료 ═══\n")
cat(sprintf("결과 폴더: %s\n", DIR_RES))
