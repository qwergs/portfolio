# ==============================================================================
# 02_모델_함수.R — 사전 Moran 스크리닝 + 3그룹 AUTO + INLA run_model
#   사전: 00_config, 01_load 가 source 됨.
#   공간객체(shp_main, g_main, nb_w)는 03_run 에서 1회 생성해 전역 보유.
#   기간 전역: CUR_YS, CUR_YE (run 단위로 설정)
# ==============================================================================

# ── 사전 Moran's I (질병 분류용) ──
moran_prescreen <- function(DAT) {
  rate_r <- DAT %>% filter(region %in% shp_main$region) %>% group_by(region) %>%
    summarise(r = sum(cases) / sum(population) * 100000, .groups = "drop")
  rvec <- rate_r$r[match(shp_main$region, rate_r$region)]; rvec[is.na(rvec)] <- 0
  if (sd(rvec, na.rm = TRUE) == 0) return(list(I = NA, p = NA, verdict = "분산0"))
  m <- tryCatch(moran.test(rvec, nb_w), error = function(e) NULL)
  if (is.null(m)) return(list(I = NA, p = NA, verdict = "오류"))
  p <- m$p.value
  list(I = unname(m$estimate[1]), p = p,
       verdict = if (p < 0.01) "★★★강함" else if (p < 0.05) "★★유의" else if (p < 0.10) "★경계" else "✗비유의",
       spatial = (p < 0.05))
}

# ── INLA M6 BYM+RW1+IID 모델 (NORO v8 run_model 일반화: 전역 cor_merged → 인자 DAT) ──
run_model <- function(TV_local, DAT, quiet = FALSE, re_str = NULL, fam = FAMILY) {
  qcat <- function(...) if (!quiet) cat(...)
  df_w <- DAT %>% filter(population > 0, region %in% shp_main$region)
  result <- list(N = nrow(df_w), n_region = n_distinct(df_w$region), n_cases = sum(df_w$cases))
  # 형태 탐색
  valid <- TV_local %>% filter(code %in% names(df_w))
  if (nrow(valid) == 0) return(result)
  form_map <- list(); data_ext <- df_w
  for (i in 1:nrow(valid)) {
    var <- valid$code[i]; x <- as.numeric(df_w[[var]])
    nv <- sum(!is.na(x) & is.finite(x)); if (nv < MIN_OBS) next
    zp <- sum(!is.na(x) & is.finite(x) & x == 0) / nv * 100; pt <- is_pct(x); hz <- zp > 20
    is_sadu <- grepl("사육두수", var); is_sex <- (var == "성비")
    if (is_sadu) { forms <- list(raw = x / 10000)
    } else if (is_sex) { forms <- list(raw = x)
    } else {
      forms <- list(raw = x)
      if (!pt) { lv <- log1p(pmax(x, 0)); lv[is.na(x)] <- NA; if (!is.na(sd(lv, na.rm = TRUE)) && sd(lv, na.rm = TRUE) > 0) forms[["log1p"]] <- lv }
      if (hz) forms[["binary"]] <- as.numeric(!is.na(x) & x > 0) else { md <- median(x, na.rm = TRUE); forms[["binary"]] <- as.numeric(!is.na(x) & x > md) }
      if (hz) { nz <- x[!is.na(x) & x > 0]; if (length(nz) > 10) { mn <- median(nz); forms[["T3"]] <- dplyr::case_when(is.na(x) ~ NA_real_, x == 0 ~ 1, x <= mn ~ 2, x > mn ~ 3) } }
      else { q33 <- quantile(x, c(1/3, 2/3), na.rm = TRUE); brk <- unique(c(-Inf, q33[1], q33[2], Inf)); if (length(brk) >= 3) forms[["T3"]] <- as.numeric(cut(x, breaks = brk, labels = FALSE, include.lowest = TRUE)) }
      q4 <- quantile(x, c(0.25, 0.5, 0.75), na.rm = TRUE); b4 <- unique(c(-Inf, q4[1], q4[2], q4[3], Inf))
      if (length(b4) >= 3) forms[["Q4"]] <- as.numeric(cut(x, breaks = b4, labels = FALSE, include.lowest = TRUE))
    }
    rr <- list(); for (fn in names(forms)) { res <- run_univ(forms[[fn]], df_w); if (!is.null(res)) rr[[fn]] <- data.frame(f = fn, p = res$p, IRR = res$IRR, n = res$n) }
    if (length(rr) == 0) next; rd <- do.call(rbind, rr) %>% arrange(p); mn_n <- floor(nv * COV_RATIO); rc <- rd[!is.na(rd$n) & rd$n >= mn_n, ]
    if (nrow(rc) == 0) rc <- rd[1, ]; bf <- rc$f[1]
    if (is_sadu) { bf <- "per10k"; bvn <- paste0(var, "__per10k"); data_ext[[bvn]] <- as.numeric(df_w[[var]]) / 10000
    } else if (is_sex) { bf <- "raw"; bvn <- var
    } else if (bf == "raw") { bvn <- var
    } else { bvn <- paste0(var, "__", bf); xcm <- as.numeric(df_w[[var]])
      if (bf == "log1p") data_ext[[bvn]] <- log1p(pmax(xcm, 0))
      else if (bf == "binary") { if (hz) data_ext[[bvn]] <- as.numeric(!is.na(xcm) & xcm > 0) else { mc <- median(xcm, na.rm = TRUE); data_ext[[bvn]] <- as.numeric(!is.na(xcm) & xcm > mc) } }
      else if (bf == "T3") { if (hz) { nzc <- xcm[!is.na(xcm) & xcm > 0]; mnc <- median(nzc, na.rm = TRUE); data_ext[[bvn]] <- dplyr::case_when(is.na(xcm) ~ NA_real_, xcm == 0 ~ 1, xcm <= mnc ~ 2, xcm > mnc ~ 3) }
        else { q33c <- quantile(xcm, c(1/3, 2/3), na.rm = TRUE); data_ext[[bvn]] <- as.numeric(cut(xcm, unique(c(-Inf, q33c[1], q33c[2], Inf)), labels = FALSE, include.lowest = TRUE)) } }
      else if (bf == "Q4") { q4c <- quantile(xcm, c(0.25, 0.5, 0.75), na.rm = TRUE); data_ext[[bvn]] <- as.numeric(cut(xcm, unique(c(-Inf, q4c[1], q4c[2], q4c[3], Inf)), labels = FALSE, include.lowest = TRUE)) }
    }
    form_map[[var]] <- list(kr = valid$kr[i], eng = valid$eng[i], cat = valid$cat[i], tier = valid$tier[i],
      forced = valid$forced[i], 이론방향 = valid$이론방향[i], 형태 = bf, 변환명 = bvn, p = rc$p[1], IRR = rc$IRR[1])
  }
  best_df <- data.frame(); for (var in names(form_map)) { m <- form_map[[var]]
    best_df <- rbind(best_df, data.frame(code = m$변환명, eng = m$eng, kr = m$kr, 형태 = m$형태, forced = m$forced,
      tier = m$tier, 이론방향 = m$이론방향, p = m$p, stringsAsFactors = FALSE)) }
  if (nrow(best_df) == 0) return(result)
  pass_vars <- best_df$code[best_df$p < PVAL_SCREEN | best_df$forced == "‡"]
  if (length(pass_vars) == 0) return(result)
  forced_c <- best_df$code[best_df$forced == "‡"]; final_vars <- pass_vars
  vif_data <- data_ext[, c("cases", final_vars), drop = FALSE]; for (v in final_vars) vif_data[[v]] <- as.numeric(vif_data[[v]])
  vif_data <- vif_data[complete.cases(vif_data), ]
  for (stp in 1:40) { if (length(final_vars) <= 1) break
    lm_t <- tryCatch(lm(as.formula(paste("cases~", paste(paste0("`", final_vars, "`"), collapse = "+"))), data = vif_data), error = function(e) NULL)
    if (is.null(lm_t)) break; vv <- tryCatch(car::vif(lm_t), error = function(e) NULL); if (is.null(vv)) break
    names(vv) <- gsub("`", "", names(vv)); if (max(vv, na.rm = TRUE) < VIF_THRESHOLD) break
    drop <- names(which.max(vv)); if (drop %in% forced_c) break; final_vars <- final_vars[final_vars != drop] }
  if (length(final_vars) == 0) return(result)
  qcat(sprintf("  INLA 투입: %d변수 (VIF<%d)\n", length(final_vars), VIF_THRESHOLD))
  FMAP <- data.frame(); for (v in final_vars) {
    m_idx <- which(sapply(form_map, function(x) x$변환명 == v)); if (length(m_idx) == 0) next; m <- form_map[[m_idx[1]]]
    x <- as.numeric(data_ext[[v]]); s <- sd(x, na.rm = TRUE); mn <- mean(x, na.rm = TRUE); safe <- paste0(m$eng, "_z")
    if (!is.na(s) && s > 0) data_ext[[safe]] <- (x - mn) / s else data_ext[[safe]] <- x
    if (m$형태 %in% c("T3","Q4")) data_ext[[paste0(m$eng, "_f")]] <- factor(as.integer(x), ordered = FALSE)
    FMAP <- rbind(FMAP, data.frame(code = v, eng = m$eng, kr = m$kr, cat = m$cat, tier = m$tier, 형태 = m$형태,
      forced = m$forced, 이론방향 = m$이론방향, safe = safe, stringsAsFactors = FALSE)) }
  if (nrow(FMAP) == 0) return(result)
  rmap <- data.frame(region = shp_main$region, idarea = seq_len(nrow(shp_main)))
  ymap <- data.frame(year = CUR_YS:CUR_YE, idtime = 1:length(CUR_YS:CUR_YE))
  ic <- data_ext[complete.cases(data_ext[, FMAP$safe]), ]
  ic <- ic %>% left_join(rmap, by = "region") %>% left_join(ymap, by = "year") %>% arrange(idarea, idtime)
  ic$idarea_time <- 1:nrow(ic)
  if (nrow(ic) < MIN_OBS) return(result)
  cov_str <- paste(FMAP$safe, collapse = " + ")
  pc_bym <- list(prec.unstruct = list(prior = "pc.prec", param = c(0.5, 0.01)), prec.spatial = list(prior = "pc.prec", param = c(0.5, 0.01)))
  pc_prec <- list(prec = list(prior = "pc.prec", param = c(0.5, 0.01)))
  if (is.null(re_str)) re_str <- "+ f(idarea,model='bym',graph=g_main,scale.model=TRUE,hyper=pc_bym)+ f(idtime,model='rw1',hyper=pc_prec)+ f(idarea_time,model='iid',hyper=pc_prec)"
  fit <- tryCatch(inla(as.formula(paste("cases ~", cov_str, "+ offset(log(population+1))", re_str)),
    family = fam, data = ic, control.compute = list(dic = TRUE, waic = TRUE, cpo = TRUE), control.predictor = list(link = 1),
    verbose = FALSE), error = function(e) NULL)
  if (is.null(fit) || is.na(fit$dic$dic)) return(result)
  qcat(sprintf("  M6 BYM+RW1+IID (NB): DIC=%.2f | N=%d\n", fit$dic$dic, nrow(ic)))
  result$dic <- fit$dic$dic; result$waic <- fit$waic$waic; result$N_final <- nrow(ic)
  result$fit <- fit; result$ic <- ic; result$FMAP <- FMAP; result$form_map <- form_map; result$data_ext <- data_ext
  fe <- fit$summary.fixed; fe <- fe[rownames(fe) != "(Intercept)", , drop = FALSE]
  if (nrow(fe) == 0) { result$n_fwd <- 0; result$n_rev <- 0; result$n_neu <- 0; result$sig_count <- 0; result$mv <- data.frame(); return(result) }
  mv <- data.frame(); sig_count <- 0
  for (k in 1:nrow(fe)) {
    irr <- round(exp(fe$mean[k]), 4); lo <- round(exp(fe$`0.025quant`[k]), 4); hi <- round(exp(fe$`0.975quant`[k]), 4)
    sig <- ifelse(fe$`0.025quant`[k] > 0 | fe$`0.975quant`[k] < 0, "★", "")
    if (hi > 1000 || lo < 0.001) sig <- ""
    rn <- gsub("_z$", "", rownames(fe)[k]); kr_n <- rn; form_str <- ""; tier_str <- ""; cat_str <- ""; theory_dir <- ""
    for (ii in 1:nrow(FMAP)) if (FMAP$eng[ii] == rn) { kr_n <- FMAP$kr[ii]; form_str <- FMAP$형태[ii]; tier_str <- FMAP$tier[ii]; cat_str <- FMAP$cat[ii]; theory_dir <- FMAP$이론방향[ii]; break }
    obs_dir <- ifelse(irr > 1, "위험↑", "보호↓")
    if (sig != "★") direction_match <- "비유의" else if (theory_dir == "중립") direction_match <- "중립"
    else { theory_risk <- (theory_dir == "위험"); obs_risk <- (irr > 1); direction_match <- ifelse(theory_risk == obs_risk, "✅정방향", "❌역방향") }
    if (sig == "★") sig_count <- sig_count + 1
    mv <- rbind(mv, data.frame(tier = tier_str, 카테고리 = cat_str, var_kr = kr_n, 형태 = form_str, IRR = irr, lo = lo, hi = hi,
      sig = sig, obs_dir = obs_dir, theory_dir = theory_dir, 방향일치 = direction_match, stringsAsFactors = FALSE))
  }
  result$mv <- mv; result$sig_count <- sig_count
  result$n_fwd <- sum(mv$방향일치 == "✅정방향"); result$n_rev <- sum(mv$방향일치 == "❌역방향")
  result$n_neu <- sum(mv$방향일치 == "중립" & mv$sig == "★")
  result
}

# ── 후보 풀 생성 (공통: cor_merged 전 컬럼 → 이론방향 배정) ──
build_candidate_pool <- function(DAT, used_codes, removed_codes = character(0)) {
  meta_cols <- c("region","year","cases","population","rate_100k","disease","area_2","area_3")
  all_cols <- setdiff(names(DAT), meta_cols)
  cand <- c()
  for (col in all_cols) {
    if (col %in% used_codes || col %in% removed_codes) next
    x <- suppressWarnings(as.numeric(DAT[[col]])); nv <- sum(!is.na(x) & is.finite(x))
    if (nv < MIN_OBS * 5) next
    sdv <- sd(x, na.rm = TRUE); if (is.na(sdv) || sdv == 0) next
    cand <- c(cand, col)
  }
  pool <- data.frame()
  for (i in seq_along(cand)) { code <- cand[i]
    pool <- rbind(pool, data.frame(tier = "AUTO", cat = assign_category(code), kr = code, code = code,
      eng = paste0("auto_", gsub("[^a-zA-Z0-9]", "_", substr(code, 1, 20)), "_", i),
      forced = "", 이론방향 = assign_theory_dir(code), stringsAsFactors = FALSE)) }
  pool
}

# ── 3그룹 동시 실행 ──
run_all_groups <- function(tv, DAT_all, DAT_urban, DAT_rural) {
  list(all   = tryCatch(run_model(tv, DAT_all,   quiet = TRUE), error = function(e) NULL),
       urban = tryCatch(run_model(tv, DAT_urban, quiet = TRUE), error = function(e) NULL),
       rural = tryCatch(run_model(tv, DAT_rural, quiet = TRUE), error = function(e) NULL))
}
get_metrics <- function(res) {
  g <- function(x, f) ifelse(is.null(x[[f]]), 0, x[[f]])
  fwd <- c(all = g(res$all,"n_fwd"), urban = g(res$urban,"n_fwd"), rural = g(res$rural,"n_fwd"))
  rev <- c(all = g(res$all,"n_rev"), urban = g(res$urban,"n_rev"), rural = g(res$rural,"n_rev"))
  list(fwd = fwd, rev = rev, total_fwd = sum(fwd), total_rev = sum(rev),
       targets_met = (fwd["all"] >= TARGET_FWD_ALL & fwd["urban"] >= TARGET_FWD_URBAN & fwd["rural"] >= TARGET_FWD_RURAL & sum(rev) == 0))
}

# ── AUTO 최적화 (Phase1 역방향 제거 → Phase2 전진 선택) ──
auto_optimize <- function(base_tv, DAT_all, DAT_urban, DAT_rural, pool) {
  cur <- base_tv
  res0 <- run_all_groups(cur, DAT_all, DAT_urban, DAT_rural); m0 <- get_metrics(res0); best_m <- m0
  # Phase 1: 역방향 제거
  for (it in 1:20) {
    if (m0$total_rev == 0) break
    all_rev <- c()
    for (nm in c("all","urban","rural")) { r <- res0[[nm]]; if (!is.null(r$mv)) { rr <- r$mv[r$mv$방향일치 == "❌역방향", ]; if (nrow(rr) > 0) all_rev <- c(all_rev, rr$var_kr) } }
    all_rev <- unique(all_rev); if (length(all_rev) == 0) break
    cur <- cur[cur$kr != all_rev[1], ]
    res0 <- run_all_groups(cur, DAT_all, DAT_urban, DAT_rural); m0 <- get_metrics(res0)
  }
  best_m <- m0
  # Phase 2: 전진 선택
  CAND <- rbind(pool %>% filter(이론방향 != "중립") %>% arrange(이론방향), pool %>% filter(이론방향 == "중립"))
  tested <- 0
  for (ci in 1:nrow(CAND)) {
    if (best_m$targets_met || tested >= MAX_ITER) break
    cand <- CAND[ci, , drop = FALSE]
    if (cand$code %in% cur$code) next
    if (!cand$code %in% names(DAT_all)) next
    tested <- tested + 1
    test_tv <- rbind(cur, cand)
    rt <- tryCatch(run_all_groups(test_tv, DAT_all, DAT_urban, DAT_rural), error = function(e) NULL)
    if (is.null(rt)) next
    mt <- get_metrics(rt)
    if (mt$total_rev == 0 && mt$total_fwd > best_m$total_fwd) { cur <- test_tv; best_m <- mt; res0 <- rt }
  }
  list(tv = cur, metrics = best_m, res = res0)
}

cat("✅ 02_모델_함수.R 로드 완료\n")
