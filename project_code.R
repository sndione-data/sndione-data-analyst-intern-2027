# NAME: Serigne N'Dione | NUMBER: 1290
# OKC Thunder  - Analyst Intern Technical Project 2027

# ── 0: Setup & Data Loading ───────────────────────────────────────────────────

library(readr)
library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(xgboost)

train <- read_csv("training.csv.gz", show_col_types = FALSE)
test  <- read_csv("testing.csv.gz",  show_col_types = FALSE)


# ── 1: Data Cleaning ──────────────────────────────────────────────────────────

# Winsorize threshold fit on training data only, applied to both sets
spd_p99 <- quantile(train$shooterspeed, 0.99, na.rm = TRUE)

train <- train %>%
  mutate(
    outcome        = as.integer(outcome),
    #  Flag tracking spikes and long-distance desperation chucks.
    is_heave       = as.integer(shottype == "heave" | distance > 47),
    # Hard caps to eliminate anomalies.
    distance       = round(pmin(distance, 47), 2),
    closestdefdist = round(pmin(closestdefdist, 35), 2),
    shooterspeed   = round(pmin(shooterspeed, spd_p99), 3),
    dribblesbefore = pmin(as.integer(dribblesbefore), 20L),
    distcont1      = round(distcont1, 2),
    distcont2      = round(distcont2, 2),
    distcont3      = round(distcont3, 2)
  )

test <- test %>%
  mutate(
    is_heave       = as.integer(shottype == "heave" | distance > 47),
    distance       = round(pmin(distance, 47), 2),
    closestdefdist = round(pmin(closestdefdist, 35), 2),
    shooterspeed   = round(pmin(shooterspeed, spd_p99), 3),
    dribblesbefore = pmin(as.integer(dribblesbefore), 20L),
    distcont1      = round(distcont1, 2),
    distcont2      = round(distcont2, 2),
    distcont3      = round(distcont3, 2)
  )

# Map hashed string identifiers to sequential integers.
# Fit strictly on the training space. Unseen test IDs receive an NA default value.
id_cols <- c("season_id", "off_team_id", "def_team_id", "shooter_id",
             "closestdef_id", "contester1_id", "contester2_id",
             "contester3_id", "contester4_id")

id_maps <- list()
for (col in id_cols) {
  lvls <- unique(na.omit(train[[col]]))
  id_maps[[col]] <- setNames(seq_along(lvls), lvls)
  train[[col]] <- id_maps[[col]][train[[col]]]
  test[[col]]  <- id_maps[[col]][test[[col]]]
  train[[col]] <- as.integer(train[[col]])
  test[[col]]  <- as.integer(test[[col]])
}


# ── 2: Time Pressure - Option Decay Index (ODI) ──────────────────────────

# When offense has time, it can present an idealized version of itself  -
#   disguise intention, survey options, preserve multiple reads to manipulate
#   the defense. Under pressure it narrows: intentions become observable,
#   players react instinctively. Shotclock is the continuous proxy for this
#   spectrum. Binned into four buckets from `shotclock`.

odi_fn <- function(shotclock) {
  case_when(
    shotclock >= 18 ~ "IDEALIZED_AMBIGUOUS",
    shotclock >=  8 ~ "PROBING_CONDITIONAL",
    shotclock >=  4 ~ "NARROWED_PREDICTABLE",
    TRUE            ~ "PANIC_INSTINCTIVE"
  )
}

train <- train %>% mutate(option_decay_index = odi_fn(shotclock))
test  <- test  %>% mutate(option_decay_index = odi_fn(shotclock))

#  Build reusable plotting tools.
make_rate_plot <- function(df, col, title, neutral_vals = NULL) {
  dat <- df %>%
    group_by(.data[[col]]) %>%
    summarise(make_rate = mean(outcome), n = n(), .groups = "drop") %>%
    mutate(label = paste0(round(make_rate * 100, 1), "%\n(n=", scales::comma(n), ")"))

  if (!is.null(neutral_vals)) {
    pal     <- colorRampPalette(c("#d73027","#f46d43","#fdae61","#fee08b","#d9ef8b","#a6d96a","#1a9850"))
    active  <- dat %>% filter(!.data[[col]] %in% neutral_vals) %>% arrange(make_rate)
    colors  <- setNames(pal(nrow(active)), active[[col]])
    dat     <- dat %>%
      mutate(bar_color = if_else(.data[[col]] %in% neutral_vals, "#b0b0b0", colors[.data[[col]]]))
    ggplot(dat, aes(x = reorder(.data[[col]], make_rate), y = make_rate, fill = bar_color)) +
      geom_col(width = 0.7) +
      geom_text(aes(label = label), hjust = -0.1, size = 3) +
      scale_fill_identity() +
      scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
      coord_flip() +
      labs(title = title, x = NULL, y = "Make Rate") +
      theme_minimal(base_size = 12)
  } else {
    ggplot(dat, aes(x = reorder(.data[[col]], make_rate), y = make_rate, fill = make_rate)) +
      geom_col(width = 0.7) +
      geom_text(aes(label = label), hjust = -0.1, size = 3) +
      scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
      scale_fill_gradientn(
        colors = c("#d73027","#f46d43","#fdae61","#fee08b","#d9ef8b","#a6d96a","#1a9850"),
        guide  = "none"
      ) +
      coord_flip() +
      labs(title = title, x = NULL, y = "Make Rate") +
      theme_minimal(base_size = 12)
  }
}

dist_plot <- function(df, col, title) {
  df %>%
    count(.data[[col]]) %>%
    mutate(pct = n / sum(n)) %>%
    ggplot(aes(x = reorder(.data[[col]], n), y = pct)) +
    geom_col(fill = "#2c7bb6", width = 0.7) +
    geom_text(aes(label = scales::percent(pct, accuracy = 0.1)), hjust = -0.1, size = 3) +
    scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
    coord_flip() +
    labs(title = title, x = NULL, y = "Share of Shots") +
    theme_minimal(base_size = 12)
}

make_rate_plot(train, "option_decay_index", "Make Rate by Option Decay Index")
dist_plot(train, "option_decay_index", "Shot Distribution - Option Decay Index")


# ── 3: Spatial Context ────────────────────────────────────────────────────────

# Shot zones define the defensive logic each attempt confronts. Derived from
#   `distance` and the `three` flag.
#
# Rim (≤5 ft): the highest point of defensive density where primary rim protection
#   and low-man helper rotations collapse.
#
# Short Mid-Range (≤14 ft, non-three): space between a backpedaling
#   drop-big and a trailing POA defender. Often optimized for slight drift drives and sudden
#   braking; where the ball has to release before reaching the rim protector's vertical contest
#   radius.
#
# Long Mid-Range (>14 ft, non-three): space willingly conceded during iso
#   switch-hunting or late-clock bailouts. Help defenders stay stationed on
#   the perimeter shell.
#
# Three: C&S kickouts from wide drives or paint collapses, and above-the-break
#   pull-ups by floor-mappers pulling rim protectors out of drop.

shot_zone_fn <- function(distance, three) {
  case_when(
    distance <= 5           ~ "rim",
    !three & distance <= 14 ~ "smr",
    !three                  ~ "lmr",
    TRUE                    ~ "three"
  )
}

train <- train %>% mutate(shot_zone = shot_zone_fn(distance, three))
test  <- test  %>% mutate(shot_zone = shot_zone_fn(distance, three))

train %>%
  group_by(shot_zone) %>%
  summarise(make_rate = round(mean(outcome), 3), n = n(), .groups = "drop") %>%
  arrange(desc(make_rate)) %>%
  print()

# Zone-relative distance: how deep within its zone a shot is attempted.
#   Rim shots near the backboard are easier than rim shots at 5 ft; corner threes
#   sit at the zone floor while above-the-break threes push toward the boundary.
#   Normalizing by zone width gives the model a clean within-zone difficulty axis.
zone_dist_rel_fn <- function(distance, shot_zone) {
  case_when(
    shot_zone == "rim"   ~ distance / 5,
    shot_zone == "smr"   ~ (distance - 5)  / 9,
    shot_zone == "lmr"   ~ (distance - 14) / 8,
    shot_zone == "three" ~ (distance - 22) / 8
  )
}
train <- train %>% mutate(zone_dist_rel = zone_dist_rel_fn(distance, shot_zone))
test  <- test  %>% mutate(zone_dist_rel = zone_dist_rel_fn(distance, shot_zone))


# ── 3b: Drive Geometry ────────────────────────────────────────────────────────

# Driving styles carry tradeoffs. The Line Winner maintains low center of
#   gravity and direct shoulder contact - driving through hip/chest without
#   deflection. When a driver lacks blow-by or absorbs chest contact, the
#   drive bends outward. Slight Drift preserves most of the rim angle; Wide
#   Driver has been fully pushed to a lateral plane, conditioning floaters,
#   decelerations, and short mid-range touch over time - but losing the macro
#   rim-pressure war, rarely reaching the front rim consistently. Perimeter
#   Release sits outside the paint geometry: shot direction, not drive angle,
#   governs. Measured from absolute lateral displacement at shot release
#   (`locationy`) and `distance`.

rim_vector_fn <- function(distance, abs_y) {
  case_when(
    distance >= 10 ~ "PERIMETER_RELEASE",
    abs_y <= 2.0   ~ "LINE_WINNER",
    abs_y <= 5.0   ~ "SLIGHT_DRIFT",
    TRUE           ~ "WIDE_DRIVER"
  )
}

train <- train %>% mutate(drive_geometry = rim_vector_fn(distance, abs(locationy)))
test  <- test  %>% mutate(drive_geometry = rim_vector_fn(distance, abs(locationy)))

train %>%
  group_by(drive_geometry) %>%
  summarise(make_rate = round(mean(outcome), 3), n = n(), .groups = "drop") %>%
  arrange(desc(make_rate)) %>%
  print()

# Lateral displacement: continuous version of drive geometry for the model.
#   abs(locationy) gives the model a raw spatial axis — a line winner at 0.5 ft
#   is meaningfully different from one at 1.8 ft even within the same category.
train <- train %>% mutate(lateral_displacement = abs(locationy))
test  <- test  %>% mutate(lateral_displacement = abs(locationy))


# ── 4: Defensive Shell Penetration ───────────────────────────────────────────

# A defensive shell collapses as offensive penetration depth increases. Synced and timed cuts
#   preserve forward momentum  - catch, gather, and finish happen in one motion,
#   giving the shell 0.5 seconds to process, organize, and react. A
#   catch-and-shoot bypasses it entirely. When a defense digs or stunts high,
#   the drive is interrupted one dribble in, leaving the shell intact. Settling
#   for mid-range creation partially engages it, but without the penetration
#   depth to collapse. Deep dribble entry triggers panic rotations before the
#   gather  - but if the handle rises too high, the player is forced into an
#   early pickup twenty feet out against a fully set defense.
#   Derived from `dribblesbefore`, `distance`, and `three`.

shell_corruption_fn <- function(drb, dist, three) {
  case_when(
    drb == 0 & dist < 8            ~ "SYNC_CUT_FINISH",
    drb == 0                       ~ "CATCH_RELEASE",
    drb <= 1 & dist >= 16 & !three ~ "FORCED_STERILE",
    drb <= 2 & dist < 6            ~ "SHALLOW_PROBE",
    drb >= 3 & dist < 10           ~ "DEEP_CORRUPTION",
    TRUE                           ~ "MID_CREATION"
  )
}

train <- train %>%
  mutate(shell_corruption = shell_corruption_fn(dribblesbefore, distance, three))
test <- test %>%
  mutate(shell_corruption = shell_corruption_fn(dribblesbefore, distance, three))

make_rate_plot(train, "shell_corruption", "Make Rate by Shell Corruption")
dist_plot(train, "shell_corruption", "Shot Distribution - Shell Corruption")


# ── 5: Lane Congestion ────────────────────────────────────────────────────────

# Lane congestion measures the defensive footprint at the exact moment a shot
#   is initiated, quantified by the number of active defenders in the lane.
#   A clear lane is entirely vacant of secondary or primary help. Single
#   coverage means one defender occupies the lane, typically the POA defender
#   recovering from behind or a static drop-big anchored in the paint. Shell
#   forming means two defenders have collapsed in, the primary plus a first-tier
#   low-man or stunt helper. Shell locked is full defensive collapse: three or
#   more defenders have formed a shell.
#   Measured directly from `num_contesters`.

lane_congestion_fn <- function(n) {
  case_when(
    n == 0 ~ "CLEAR_LANE",
    n == 1 ~ "SINGLE_COVERAGE",
    n == 2 ~ "SHELL_FORMING",
    TRUE   ~ "SHELL_LOCKED"
  )
}

train <- train %>% mutate(lane_congestion = lane_congestion_fn(num_contesters))
test  <- test  %>% mutate(lane_congestion = lane_congestion_fn(num_contesters))

make_rate_plot(train, "lane_congestion", "Make Rate by Lane Congestion")

# Ordinal encoding: continuous defensive pressure gradient for the model.
#   3 = clear lane, 2 = single coverage, 1 = shell forming, 0 = shell locked.
train <- train %>% mutate(lane_cong_ord = case_when(
  num_contesters == 0 ~ 3L,
  num_contesters == 1 ~ 2L,
  num_contesters == 2 ~ 1L,
  TRUE                ~ 0L
))
test <- test %>% mutate(lane_cong_ord = case_when(
  num_contesters == 0 ~ 3L,
  num_contesters == 1 ~ 2L,
  num_contesters == 2 ~ 1L,
  TRUE                ~ 0L
))


# ── 6: Player Identity  - Zone Profiling ───────────────────────────────────────

# To measure a player's true efficiency in any given spot on the floor, we
#   measure their performance against the baseline of the field. A leave-one-out
#   calculation removes the current attempt from the equation entirely  -
#   preventing a single shot from distorting its own evaluation. The resulting
#   gap between the player's zone baseline and the league standard determines
#   their classification. Only the elite, those tracking in the top ten to
#   eighteen percent of all shooters, receive the highest classification in a
#   zone. Those with fewer than ten attempts remain unmeasured, defaulting
#   safely to the neutral tier.
#   Measured from `shooter_id` grouped FG% by zone, leave-one-out.
#
# Zone baselines computed from training data
ZONE_AVG <- train %>%
  group_by(shot_zone) %>%
  summarise(baseline = mean(outcome), .groups = "drop") %>%
  tibble::deframe()
TIP_AVG <- mean(train$outcome[train$shottype == "tip"])

print(ZONE_AVG)
cat("Tip baseline:", round(TIP_AVG, 3), "\n")

# LOO encoding per zone - computed on training data only to prevent future leakage
for (z in c("rim", "smr", "lmr", "three")) {
  col_name <- paste0(z, "_loo")
  zone_stats <- train %>%
    filter(shot_zone == z) %>%
    group_by(shooter_id) %>%
    mutate(
      s_sum    = sum(outcome),
      s_count  = n(),
      # Sample size barrier: players must log >10 attempts to unlock profiling rows.
      !!col_name := if_else(s_count > 10, (s_sum - outcome) / (s_count - 1), NA_real_)
    ) %>%
    ungroup() %>%
    select(shot_id, all_of(col_name))
  train <- train %>% left_join(zone_stats, by = "shot_id")
}

# Profile construction and Delta distribution analysis
shooter_profiles_raw <- train %>%
  group_by(shooter_id) %>%
  summarise(
    total_shots = n(),
    rim_delta   = if_else(sum(shot_zone == "rim")   > 10,
                    mean(rim_loo[shot_zone == "rim"],     na.rm = TRUE) - ZONE_AVG["rim"],   NA_real_),
    smr_delta   = if_else(sum(shot_zone == "smr")   > 10,
                    mean(smr_loo[shot_zone == "smr"],     na.rm = TRUE) - ZONE_AVG["smr"],   NA_real_),
    lmr_delta   = if_else(sum(shot_zone == "lmr")   > 10,
                    mean(lmr_loo[shot_zone == "lmr"],     na.rm = TRUE) - ZONE_AVG["lmr"],   NA_real_),
    three_delta = if_else(sum(shot_zone == "three") > 10,
                    mean(three_loo[shot_zone == "three"], na.rm = TRUE) - ZONE_AVG["three"], NA_real_),
    tip_delta   = if_else(sum(shottype == "tip") >= 5,
                    mean(outcome[shottype == "tip"]) - TIP_AVG, NA_real_),
    .groups = "drop"
  )

# Explore delta distributions - thresholds drawn from these shapes
delta_hist <- function(df, col, zone_label) {
  df %>% filter(!is.na(.data[[col]])) %>%
    ggplot(aes(x = .data[[col]])) +
    geom_histogram(bins = 40, fill = "#2c7bb6", color = "white") +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray40") +
    labs(title = paste("LOO Delta Distribution:", zone_label),
         x = "Delta vs Zone Baseline", y = "Shooters") +
    theme_minimal(base_size = 12)
}

delta_hist(shooter_profiles_raw, "rim_delta",   "Rim")
delta_hist(shooter_profiles_raw, "smr_delta",   "Short Mid-Range")
delta_hist(shooter_profiles_raw, "lmr_delta",   "Long Mid-Range")
delta_hist(shooter_profiles_raw, "three_delta", "Three")
delta_hist(shooter_profiles_raw, "tip_delta",   "Tip / Putback")

# Threshold Assignment Functions derived from empirical distribution shapes
rim_profile_fn <- function(d) {
  case_when(
    d >= .08             ~ "RIM_COLLAPSE_FORCE",
    d >= .02             ~ "CONTEST_RESISTANT",
    is.na(d) | d > -.02 ~ "LEVERAGE_NEUTRAL",
    TRUE                 ~ "RIM_TIMID"
  )
}
smr_profile_fn <- function(d) {
  case_when(
    d >= .05 ~ "DROP_COVERAGE_KILLER",
    d >= .01 ~ "PROBING_TOUCH",
    TRUE     ~ "REACTIVE_BAILOUT"
  )
}
lmr_profile_fn <- function(d) {
  case_when(
    d >= .06 ~ "MIDRANGE_WEAPON",
    d >= .02 ~ "STRETCH_COUNTER",
    TRUE     ~ "INEFFICIENT_DRIFT"
  )
}
three_profile_fn <- function(d) {
  case_when(
    d >= .04  ~ "ELITE_THREAT",
    d >= .015 ~ "SOLID_SPACER",
    TRUE      ~ "LOW_PRIORITY"
  )
}
tip_profile_fn <- function(d) {
  case_when(
    d >= .10              ~ "AERIAL_FINISHER",
    is.na(d) | d >= -.10 ~ "OPPORTUNISTIC_TIP",
    TRUE                  ~ "CRASH_INEFFICIENT"
  )
}

# ── 6b: Offensive Identity ───────────────────────────────────────────────────

# Zone efficiency labels combine into a shooter zone profile and an attacking
#   profile. The attacking profile classifies frequency patterns using dynamic
#   percentile thresholds fit on shooters with at least 30 attempts, so
#   classifications reflect the actual distribution of the dataset rather than
#   fixed cutoffs.

shooter_profiles <- shooter_profiles_raw %>%
  mutate(
    rim_profile   = rim_profile_fn(rim_delta),
    smr_profile   = smr_profile_fn(smr_delta),
    lmr_profile   = lmr_profile_fn(lmr_delta),
    three_profile = three_profile_fn(three_delta),
    tip_profile   = tip_profile_fn(tip_delta)
  )

# Zone share frequencies per shooter
shooter_shares <- train %>%
  group_by(shooter_id) %>%
  summarise(
    total_shots = n(),
    rim_share   = mean(shot_zone == "rim"),
    smr_share   = mean(shot_zone == "smr"),
    lmr_share   = mean(shot_zone == "lmr"),
    three_share = mean(shot_zone == "three"),
    outer_share = mean(shot_zone %in% c("lmr", "three")),
    .groups = "drop"
  )

shooter_profiles <- shooter_profiles %>%
  left_join(shooter_shares %>% select(-total_shots), by = "shooter_id")

# Percentiles are calibrated strictly on high-conviction players (>=30 shots)
q30 <- shooter_profiles %>% filter(total_shots >= 30)
RIM_P65   <- quantile(q30$rim_share,   0.65)
RIM_P50   <- quantile(q30$rim_share,   0.50)
OUTER_P65 <- quantile(q30$outer_share, 0.65)
OUTER_P50 <- quantile(q30$outer_share, 0.50)
OUTER_P40 <- quantile(q30$outer_share, 0.40)
SMR_P65   <- quantile(q30$smr_share,   0.65)

shooter_profiles <- shooter_profiles %>%
  mutate(
    outer_eff = (!is.na(lmr_delta) & lmr_delta >= 0) | (!is.na(three_delta) & three_delta >= 0),
    rim_eff   = !is.na(rim_delta) & rim_delta >= 0,
    attacking_profile = case_when(
      # Everywhere Scorer: High-volume plus verified positive efficiency across multiple sectors.
      rim_share >= RIM_P50 & outer_share >= OUTER_P50 & rim_eff & outer_eff ~ "EVERYWHERE_SCORER",
      # Players who survive as shotmakers on the boundary of the shell - not having to live on attacking where the defense is most dense.
      outer_share >= OUTER_P65 & outer_eff                                  ~ "BOUNDARY_SHELL_BYPASSER",
      rim_share >= RIM_P65 & outer_share <= OUTER_P40                       ~ "CENTRAL_RIM_DEPENDENT",
      smr_share >= SMR_P65 & !is.na(smr_delta) & smr_delta >= 0            ~ "INTERMEDIATE_GAP_EXPLOITER",
      TRUE                                                                   ~ "CONDITIONAL_SYSTEM_SCORER"
    )
  )

profile_cols <- c("shooter_id", "attacking_profile", "rim_profile", "smr_profile",
                  "lmr_profile", "three_profile", "tip_profile")

train <- train %>% left_join(shooter_profiles[, profile_cols], by = "shooter_id")
test  <- test  %>% left_join(shooter_profiles[, profile_cols], by = "shooter_id")

# Isolate the single relevant zone profile per shot row location.
train <- train %>%
  mutate(shooter_zone_profile = case_when(
    shot_zone == "rim"   ~ rim_profile,
    shot_zone == "smr"   ~ smr_profile,
    shot_zone == "lmr"   ~ lmr_profile,
    shot_zone == "three" ~ three_profile
  ))
test <- test %>%
  mutate(shooter_zone_profile = case_when(
    shot_zone == "rim"   ~ rim_profile,
    shot_zone == "smr"   ~ smr_profile,
    shot_zone == "lmr"   ~ lmr_profile,
    shot_zone == "three" ~ three_profile
  ))

for (z in c("rim", "smr", "lmr", "three")) {
  cat("\n--", toupper(z), "--\n")
  train %>%
    filter(shot_zone == z) %>%
    group_by(shooter_zone_profile) %>%
    summarise(make_rate = round(mean(outcome), 3), n = n(), .groups = "drop") %>%
    arrange(desc(make_rate)) %>%
    print()
}

train %>%
  group_by(attacking_profile) %>%
  summarise(make_rate = round(mean(outcome), 3), n = n(), .groups = "drop") %>%
  arrange(desc(make_rate)) %>%
  print()


# ── 7: Defensive Identity  - Defender Role ─────────────────────────────────────

# Defensive identity is inferred from shot location  - specifically where, on
#   the floor, a defender most frequently contests shots. A high rim rate
#   combined with minimal three-point exposure identifies an anchor big:
#   paint-bound, rarely leaving the restricted area. A high rim rate with
#   broader range identifies a mobile big, capable of showing on perimeter
#   possessions. A high three rate with a low rim rate identifies a perimeter
#   stopper  - a point-of-attack defender living on the arc. Everything else
#   is a balanced rotator: scheme-dependent, help-side, no dominant zone.
#   Defenders with fewer than thirty shots defended default to balanced rotator.
#   Inferred from the shot zone of every attempt where the player appears
#   as `closestdef_id`.

defender_profiles <- train %>%
  group_by(closestdef_id) %>%
  summarise(
    total_def  = n(),
    rim_rate   = mean(shot_zone == "rim"),
    three_rate = mean(shot_zone == "three"),
    .groups    = "drop"
  ) %>%
  filter(total_def >= 30) %>%
  # Zero contest handler: Maps empty perimeter portfolios safely to Infinity to prevent NaN crashes.
  mutate(rim_three_ratio = if_else(three_rate > 0, rim_rate / three_rate, Inf))

DEF_RIM_P80   <- quantile(defender_profiles$rim_rate,        0.80)
DEF_RIM_P65   <- quantile(defender_profiles$rim_rate,        0.65)
DEF_RIM_P50   <- quantile(defender_profiles$rim_rate,        0.50)
DEF_THREE_P80 <- quantile(defender_profiles$three_rate,      0.80)
DEF_RATIO_P75 <- quantile(
  defender_profiles$rim_three_ratio[is.finite(defender_profiles$rim_three_ratio)], 0.75)

defender_profiles <- defender_profiles %>%
  mutate(defender_role = case_when(
    # Anchor Big: Heavy rim protection and deep restricted area anchoring with minimal perimeter drift.
    rim_rate >= DEF_RIM_P80 & rim_three_ratio >= DEF_RATIO_P75 ~ "ANCHOR_BIG",
    rim_rate >= DEF_RIM_P65                                    ~ "MOBILE_BIG",
    # Perimeter Stopper: Point-of-attack perimeter specialists locked onto the perimeter arc.
    three_rate >= DEF_THREE_P80 & rim_rate <= DEF_RIM_P50      ~ "PERIMETER_STOPPER",
    TRUE                                                       ~ "BALANCED_ROTATOR"
  ))

train <- train %>%
  left_join(defender_profiles[, c("closestdef_id", "defender_role")], by = "closestdef_id") %>%
  mutate(defender_role = replace_na(defender_role, "BALANCED_ROTATOR"))
test <- test %>%
  left_join(defender_profiles[, c("closestdef_id", "defender_role")], by = "closestdef_id") %>%
  mutate(defender_role = replace_na(defender_role, "BALANCED_ROTATOR"))

train %>%
  group_by(defender_role) %>%
  summarise(make_rate = round(mean(outcome), 3), n = n(), .groups = "drop") %>%
  arrange(desc(make_rate)) %>%
  print()
dist_plot(train, "defender_role", "Shot Distribution - Defender Role")

# Defender zone fit: whether the defender is operating in their optimal or hostile
#   zone. An anchor big at the rim is OPTIMAL; dragged to the three-point line is
#   HOSTILE. A perimeter stopper at the arc is OPTIMAL; caught on a rim drive is
#   HOSTILE. Everything else is NEUTRAL.
defender_zone_fit_fn <- function(role, zone) {
  case_when(
    role == "ANCHOR_BIG"        & zone %in% c("rim", "smr") ~ "OPTIMAL",
    role == "ANCHOR_BIG"        & zone == "three"            ~ "HOSTILE",
    role == "MOBILE_BIG"        & zone == "rim"              ~ "OPTIMAL",
    role == "MOBILE_BIG"        & zone == "three"            ~ "HOSTILE",
    role == "PERIMETER_STOPPER" & zone %in% c("lmr","three") ~ "OPTIMAL",
    role == "PERIMETER_STOPPER" & zone == "rim"              ~ "HOSTILE",
    TRUE                                                     ~ "NEUTRAL"
  )
}
train <- train %>% mutate(defender_zone_fit = defender_zone_fit_fn(defender_role, shot_zone))
test  <- test  %>% mutate(defender_zone_fit = defender_zone_fit_fn(defender_role, shot_zone))

train %>%
  group_by(defender_zone_fit) %>%
  summarise(make_rate = round(mean(outcome), 3), n = n(), .groups = "drop") %>%
  arrange(desc(make_rate)) %>%
  print()

# ── 8: Environmental Quality – Shot Quality (SQ) ─────────────────────────────

# Shot Quality is, at its heart, a measure of the friction generated by the
#   defense against the shooter's (or team's) ability to manufacture windows of
#   space and efficiency. Naturally, it serves as a lens to reveal the burden or
#   ease a shotmaker faces on a given shot before shooting talent enters the
#   equation.

# Space (S) awards a premium to the immediate rim and perimeter over dead-zone
#   mid-range shots
train$S <- with(train, ifelse(distance <= 5, 1.2,
                  ifelse(!three & distance <= 14, 0.8,
                  ifelse(!three, 0.7, 1.05))))
test$S  <- with(test,  ifelse(distance <= 5, 1.2,
                  ifelse(!three & distance <= 14, 0.8,
                  ifelse(!three, 0.7, 1.05))))

# Time (T) holds at a completely neutral 1.00 for 85% of the possession before
#   applying a gentle discount during the final four seconds to account for the
#   internal rush of beating the buzzer.

train$T_wt <- 1.0 - (0.10 * (4.0 - pmin(train$shotclock, 4.0)) / 4.0)
test$T_wt  <- 1.0 - (0.10 * (4.0 - pmin(test$shotclock,  4.0)) / 4.0)

# Defense (D) balances the cushion of the measurement with closing velocity,
#   step-back separation, and help distance from secondary defenders.
#   Parsed from the approach time series embedded in closestdefapproach {d1..d4}.

app_train   <- strsplit(gsub("[{}]", "", train$closestdefapproach), ",")
train$d1    <- as.numeric(sapply(app_train, function(x) x[1]))
train$d3    <- as.numeric(sapply(app_train, function(x) x[3]))
train$d4    <- as.numeric(sapply(app_train, function(x) x[4]))

app_test    <- strsplit(gsub("[{}]", "", test$closestdefapproach), ",")
test$d1     <- as.numeric(sapply(app_test, function(x) x[1]))
test$d3     <- as.numeric(sapply(app_test, function(x) x[3]))
test$d4     <- as.numeric(sapply(app_test, function(x) x[4]))

# CloseoutRate = d1 - d4: A positive value denotes a hard closeout, acting as
#   an explicit penalty.
train$CloseoutRate <- train$d1 - train$d4
test$CloseoutRate  <- test$d1  - test$d4

# ReleaseSlope = d4 - d3: A positive value confirms late space opening up right
#   at elevation.
train$ReleaseSlope <- train$d4 - train$d3
test$ReleaseSlope  <- test$d4  - test$d3

train$HelpDistance <- (ifelse(is.na(train$distcont1), 25, train$distcont1) +
                       ifelse(is.na(train$distcont2), 25, train$distcont2) +
                       ifelse(is.na(train$distcont3), 25, train$distcont3)) / 3
test$HelpDistance  <- (ifelse(is.na(test$distcont1),  25, test$distcont1)  +
                       ifelse(is.na(test$distcont2),  25, test$distcont2)  +
                       ifelse(is.na(test$distcont3),  25, test$distcont3)) / 3

train$D_raw <- 0.40 * train$d4 - 0.25 * train$CloseoutRate +
               0.15 * train$ReleaseSlope + 0.20 * train$HelpDistance
test$D_raw  <- 0.40 * test$d4  - 0.25 * test$CloseoutRate  +
               0.15 * test$ReleaseSlope  + 0.20 * test$HelpDistance

# To protect the logic from collapsing in heavily contested spaces, the
#   row-level floor is applied in training before normalizing against the
#   99th percentile ceiling.
D_p99        <- quantile(pmax(train$D_raw, 0), 0.99, na.rm = TRUE)
train$D_norm <- pmin(pmax(train$D_raw, 0) / D_p99, 1.0)
test$D_norm  <- pmin(pmax(test$D_raw,  0) / D_p99, 1.0)

train$is_heave <- as.integer(train$shottype == "heave" | train$distance > 47)
test$is_heave  <- as.integer(test$shottype  == "heave" | test$distance  > 47)

train$SQ <- ifelse(train$is_heave == 1, 0,
              round((train$S * train$T_wt * train$D_norm) / 1.20, 3))
test$SQ  <- ifelse(test$is_heave  == 1, 0,
              round((test$S  * test$T_wt  * test$D_norm)  / 1.20, 3))

# Scale validation: confirm the formula is behaving before any modeling use.
cat("SQ distribution (training):\n")
print(round(quantile(train$SQ, c(0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0)), 3))
cat(sprintf("Mean: %.3f  |  SD: %.3f\n", mean(train$SQ), sd(train$SQ)))

# Heave override: all desperation heaves must return exactly 0.
cat(sprintf("\nHeave SQ check — mean: %.3f  (n=%d)\n",
  mean(train$SQ[train$is_heave == 1]),
  sum(train$is_heave == 1)))

sq_cols <- c("shottype", "distance", "shotclock", "d1", "d4",
             "CloseoutRate", "ReleaseSlope", "HelpDistance",
             "D_norm", "S", "T_wt", "SQ", "outcome")

# Dunks are handled naturally by the approach series without a manual override.
cat("\nTop dunk shots by SQ:\n")
train %>%
  filter(shottype == "dunk") %>%
  arrange(desc(SQ)) %>%
  head(3) %>%
  select(all_of(sq_cols)) %>%
  as.data.frame() %>%
  print()

# Open fast break rim: the ceiling of the formula — uncontested, clear lane,
#   immediate rim. SQ should hit 1.0 with near-perfect make rate.
cat("\nTop open fast break rim shots by SQ:\n")
train %>%
  filter(shot_zone == "rim", num_contesters == 0, dribblesbefore == 0) %>%
  arrange(desc(SQ)) %>%
  head(3) %>%
  select(all_of(sq_cols)) %>%
  as.data.frame() %>%
  print()

# Open catch-and-shoot three: formula ceiling for arc shots is 0.875.
cat("\nTop open catch-and-shoot threes by SQ:\n")
train %>%
  filter(three == TRUE, num_contesters == 0, dribblesbefore == 0) %>%
  arrange(desc(SQ)) %>%
  head(3) %>%
  select(all_of(sq_cols)) %>%
  as.data.frame() %>%
  print()

# ── Team-level SQ ─────────────────────────────────────────────────────────────

# Offensive and defensive SQ at the team level establishes that the formula
#   produces meaningful macro signal before drilling into individual variables.
#   Net differential identifies which teams are tilting shot quality in their
#   favor across a full season of possessions.

cat("\nOffensive SQ by team:\n")
train %>%
  group_by(off_team_id) %>%
  summarise(n = n(), mean_SQ = round(mean(SQ), 3),
            make_rate = round(mean(outcome), 3), .groups = "drop") %>%
  arrange(desc(mean_SQ)) %>%
  print(n = 30)

cat("\nDefensive SQ allowed by team:\n")
train %>%
  group_by(def_team_id) %>%
  summarise(n = n(), mean_SQ_allowed = round(mean(SQ), 3),
            opp_make_rate = round(mean(outcome), 3), .groups = "drop") %>%
  arrange(mean_SQ_allowed) %>%
  print(n = 30)

cat("\nNet SQ differential (off - def allowed):\n")
off_sq <- train %>%
  group_by(team = off_team_id) %>%
  summarise(off_SQ = round(mean(SQ), 3), .groups = "drop")
def_sq <- train %>%
  group_by(team = def_team_id) %>%
  summarise(def_SQ = round(mean(SQ), 3), .groups = "drop")
inner_join(off_sq, def_sq, by = "team") %>%
  mutate(net_SQ = round(off_SQ - def_SQ, 3)) %>%
  arrange(desc(net_SQ)) %>%
  print(n = 30)

cat("\nSQ by lane congestion:\n")
train %>%
  mutate(lane_congestion = case_when(
    num_contesters == 0 ~ "CLEAR_LANE",
    num_contesters == 1 ~ "SINGLE_COVERAGE",
    num_contesters == 2 ~ "SHELL_FORMING",
    TRUE                ~ "SHELL_LOCKED"
  )) %>%
  group_by(lane_congestion) %>%
  summarise(n = n(), mean_SQ = round(mean(SQ), 3),
            make_rate = round(mean(outcome), 3), .groups = "drop") %>%
  arrange(desc(mean_SQ)) %>%
  print()

cat("\nSQ by shell corruption:\n")
train %>%
  group_by(shell_corruption) %>%
  summarise(n = n(), mean_SQ = round(mean(SQ), 3),
            make_rate = round(mean(outcome), 3), .groups = "drop") %>%
  arrange(desc(mean_SQ)) %>%
  print()

# Three-point creation depth: the SQ-to-make-rate gap widens with every dribble
#   because D_norm rises as the defender retreats, but body mechanics degrade.
cat("\nThree SQ by dribble creation depth:\n")
train %>%
  filter(three == TRUE, is_heave == 0) %>%
  mutate(creation_bucket = case_when(
    dribblesbefore == 0 ~ "0_CATCH_RELEASE",
    dribblesbefore == 1 ~ "1_ONE_DRIBBLE",
    dribblesbefore == 2 ~ "2_TWO_DRIBBLE",
    dribblesbefore <= 4 ~ "3_MODERATE",
    TRUE                ~ "4_HIGH_CREATION"
  )) %>%
  group_by(creation_bucket) %>%
  summarise(n = n(), mean_SQ = round(mean(SQ), 3),
            make_rate = round(mean(outcome), 3),
            SQ_make_gap = round(mean(SQ) - mean(outcome), 3),
            mean_D_norm = round(mean(D_norm), 3),
            mean_closeout = round(mean(CloseoutRate), 3), .groups = "drop") %>%
  arrange(creation_bucket) %>%
  print()

cat("\nSQ by drive geometry:\n")
train %>%
  mutate(drive_geometry = case_when(
    distance >= 10        ~ "PERIMETER_RELEASE",
    abs(locationy) <= 2.0 ~ "LINE_WINNER",
    abs(locationy) <= 5.0 ~ "SLIGHT_DRIFT",
    TRUE                  ~ "WIDE_DRIVER"
  )) %>%
  group_by(drive_geometry) %>%
  summarise(n = n(), mean_SQ = round(mean(SQ), 3),
            make_rate = round(mean(outcome), 3), .groups = "drop") %>%
  arrange(desc(mean_SQ)) %>%
  print()

# ── SQ × Player Profiles ──────────────────────────────────────────────────────

# Crossing SQ against attacking and defender profiles tests whether the identity
#   categories built earlier align with the environmental quality signal.

cat("\nSQ by attacking profile:\n")
train %>%
  group_by(attacking_profile) %>%
  summarise(n = n(), mean_SQ = round(mean(SQ), 3),
            make_rate = round(mean(outcome), 3), .groups = "drop") %>%
  arrange(desc(mean_SQ)) %>%
  print()

cat("\nSQ allowed by defender role:\n")
train %>%
  group_by(defender_role) %>%
  summarise(n = n(), mean_SQ = round(mean(SQ), 3),
            opp_make_rate = round(mean(outcome), 3), .groups = "drop") %>%
  arrange(mean_SQ) %>%
  print()

cat("\nSQ allowed by defender role × shot zone:\n")
train %>%
  group_by(defender_role, shot_zone) %>%
  summarise(n = n(), mean_SQ = round(mean(SQ), 3),
            opp_make_rate = round(mean(outcome), 3), .groups = "drop") %>%
  arrange(shot_zone, mean_SQ) %>%
  print(n = 20)

# We like to imagine that as we obtain more and more information, we become better at understanding the meaning of that information, and therefore properly labeling
# what we're observing…

# The player does things, we observe through our fogged lens, and eventually put an accurate enough label on whatever we're observing, 
# and we're mostly content with that order of operations...but should there not be more curiosity?


# Basketball is painfully relational so our Shot Quality tests across variables interrogated some priors

# ── Feature Inventory Tests ──────────────────────────────────────────


# Lane Congestion x SQ:
# The baseline tracks the defensive shell with a near-perfect gradient from a clear_lane(0.621 SQ/ 62.1% make) down to a suffocated shell_locked(0.071 SQ/36.9% make)… 
# Closeout Velocity remains nearly flat across all four congestion tiers, proving lane congestion and closeout velocity have zero redundancy.

# But, the HelpDistance in D is doing the heavy lifting, in a clear lane it sits locked at 25 ft, under shell locked it contracts to 3.4, capturing the swarm of defense


# Shell Corruption x SQ:
# A clean inversion where a stationary catch release gives good space but lower conversion(0.455 SQ / 37.9% make) while a sync cut defies its tight
# environment(0.332 SQ / 68.4% make) because the variable can't read the rim protector's lower margin for error to turn and square up to contest


# Three Creation x SQ:
# The shot quality widens with every bounce as the defender retreats or is caught in a high Pick & Roll alignment, 
# yet actual efficiency decays because off-the-dribble pull ups inject more difficulty into the shot


# Drive Geometry x SQ:
# Line winner punishes the interior at a 63.9% clip despite an unrewarding shot quality environment, while a wide driver captures wide open cushions through
# conditioned space creation habits(stepbacks, decelerations… shown by high Release Slope) but fails to convert due to broken finishing angles & shot difficulty.


# Attacking Profile x SQ:
# A central rim dependent absorbs a ton of friction to convert tough possessions(0.355 SQ / 51.5% make), whereas players who survives as shotmakers
# outside the defense's shell have the highest quality but lowest make rate(42.2%)


# Defender Role x SQ:
# An elite anchor big acts as a premier paint suppressor (0.366 SQ allowed) but leaks severe value when dragged out to the perimeter(0.514 SQ allowed), while a perimeter stopper
# exhibits the exact inverse footprint. It would be worthwhile to make an interaction between the zone being defended and the defender's optimal/hostile defensive environment(Check)


# ── 8b: Defender Shot Quality Allowed (def_sq_loo) ───────────────────────────

# Per-defender leave-one-out average SQ allowed is a continuous signal of defender
#   quality that sits beneath the role labels. A rim anchor who routinely faces
#   high-SQ rim attempts scores differently from one who never gets tested.
#   LOO removes the current shot so no single attempt inflates its own defender's
#   profile. Defenders with ≤10 shots defended receive NA and are imputed with
#   the training mean at model time.

def_sq_profile <- train %>%
  group_by(closestdef_id) %>%
  mutate(
    def_cnt    = n(),
    sq_sum     = sum(SQ),
    # Seperate the environment from the singular event
    def_sq_loo = if_else(def_cnt > 10, (sq_sum - SQ) / (def_cnt - 1), NA_real_)
  ) %>%
  ungroup() %>%
  select(shot_id, def_sq_loo)

train <- train %>% left_join(def_sq_profile, by = "shot_id")

# Aggregate out-of-sample for test set stability
def_sq_avg <- train %>%
  filter(!is.na(def_sq_loo)) %>%
  group_by(closestdef_id) %>%
  summarise(def_sq_loo = mean(def_sq_loo), .groups = "drop")

# Project training-calibrated defensive priors onto the unseen test data
test <- test %>%
  select(-any_of("def_sq_loo")) %>%
  left_join(def_sq_avg, by = "closestdef_id")

# Verify scale health
train %>%
  filter(!is.na(def_sq_loo)) %>%
  summarise(mean = round(mean(def_sq_loo), 3),
            sd   = round(sd(def_sq_loo),   3),
            min  = round(min(def_sq_loo),  3),
            max  = round(max(def_sq_loo),  3)) %>%
  print()


# ── 9: Shot Making Efficiency (SME) ──────────────────────────────────────────

#   While Shot Quality can help measure the baseline probability of a possession,
#   they can never fully predict the finality of shot execution. To bridge this
#   gap, we add its companion piece: Shot Making Efficiency(SME)

# Empirical Bayes smoothing replaces my inital hard 10-shot volume cutoff.
#   Rather than zeroing out low-volume shooters entirely, the raw increase in shot making efficiency is  multiplied by n / (n + 60). A shooter with 5 zone attempts contributes
#   7% of their observed delta; one with 200 attempts contributes 77%. This prevents hot streaks from small samples leaking while preserving signal from high-volume shooters.
EB_K_SME <- 60

# Zone-level SQ calibration matrix — fit on training data only to lock empirical priors.
sq_calibration <- train %>%
  mutate(sq_bucket = round(SQ / 0.05) * 0.05) %>%
  group_by(shot_zone, sq_bucket) %>%
  summarise(expected_make_rate = mean(outcome), .groups = "drop")

# Join expected make rate to each shot, aggregate per shooter per zone.
player_zone_sme <- train %>%
  mutate(sq_bucket = round(SQ / 0.05) * 0.05) %>%
  left_join(sq_calibration, by = c("shot_zone", "sq_bucket")) %>%
  mutate(expected_make_rate = coalesce(expected_make_rate, 0.45)) %>%
  group_by(shooter_id, shot_zone) %>%
  summarise(
    zone_fga      = n(),
    act_make_rate = round(mean(outcome), 3),
    exp_make_rate = round(mean(expected_make_rate), 3),
    .groups = "drop"
  ) %>%
  # Apply continuous Empirical Bayes smoothing instead of a rigid threshold cliff
  mutate(SME_zone = round((act_make_rate - exp_make_rate) * zone_fga / (zone_fga + EB_K_SME), 3))

# Pivot to wide format and join back to train and test as per-shot features.
sme_wide <- player_zone_sme %>%
  select(shooter_id, shot_zone, SME_zone) %>%
  pivot_wider(names_from = shot_zone, values_from = SME_zone,
              names_prefix = "SME_", values_fill = 0)

train <- train %>% left_join(sme_wide, by = "shooter_id")
test  <- test  %>% left_join(sme_wide, by = "shooter_id")

# Per-shot active zone SME: pull the relevant zone column for each row.
train <- train %>%
  mutate(active_zone_sme = case_when(
    shot_zone == "rim"   ~ SME_rim,
    shot_zone == "smr"   ~ SME_smr,
    shot_zone == "lmr"   ~ SME_lmr,
    shot_zone == "three" ~ SME_three
  ))
test <- test %>%
  mutate(active_zone_sme = case_when(
    shot_zone == "rim"   ~ SME_rim,
    shot_zone == "smr"   ~ SME_smr,
    shot_zone == "lmr"   ~ SME_lmr,
    shot_zone == "three" ~ SME_three
  ))


# ── 10: MODEL TRAINING & SUBMISSION ──────────────────────────────────────────
# Feed an extreme gradient boosted XGBoost architecture our orthogonal features to predict results.



# ── 10a: FEATURE ARCHITECTURE ─────────────────────────────────────────
# Partition our engineered features into continuous arrays and categorical buckets

num_f <- c(
  "SQ", "D_norm", "zone_dist_rel", "lateral_displacement",
  "shooterspeed", "CloseoutRate", "ReleaseSlope", "HelpDistance",
  "lane_cong_ord", "active_zone_sme", "def_sq_loo"
)

cat_f <- c(
  "shot_zone", "shottype", "gamestate", "shell_corruption",
  "drive_geometry", "option_decay_index", "defender_role",
  "attacking_profile", "shooter_zone_profile", "defender_zone_fit"
)

# Encoding factor levels while cleaning missing elements to deliver a uniform canvas to the model

build_mat <- function(df, num_features, cat_features, ref_cols = NULL) {

  # One-hot encode categoricals to intercept unseen/missing entries safely
  for (f in cat_features) {
    df[[f]] <- as.character(df[[f]])
    df[[f]][is.na(df[[f]])] <- "UNKNOWN"
    df[[f]] <- as.factor(df[[f]])
  }

  # Continuous NA stabilization pass using fixed reference pillars to prevent data leakage
  for (f in num_features) {
    if (any(is.na(df[[f]]))) {
      fill_val <- if (!is.null(ref_cols)) ref_cols[[f]] else median(df[[f]], na.rm = TRUE)
      df[[f]][is.na(df[[f]])] <- fill_val
    }
  }

  # Expand features into a full-rank numeric map
  mat <- model.matrix(~ . - 1,
                      data = df[, c(num_features, cat_features), drop = FALSE])
  mat
}

#Freeze training parameters here to use as baselines across the test set
num_medians <- lapply(num_f, function(f) median(train[[f]], na.rm = TRUE))
names(num_medians) <- num_f

X_train <- build_mat(train, num_f, cat_f, ref_cols = num_medians)
X_test  <- build_mat(test,  num_f, cat_f, ref_cols = num_medians)

# Out-of-sample alignment sweep
missing_cols <- setdiff(colnames(X_train), colnames(X_test))
for (col in missing_cols) X_test <- cbind(X_test, setNames(data.frame(0), col))
X_test <- X_test[, colnames(X_train), drop = FALSE]

y_train <- train$outcome


# ── 10b: XGBOOST REGULARIZED LEARNING ────────────────────────────────────────

# Transform features into optimized arrays to accelerate training. The engine learns incrementally
# at a deliberate, low learning rate (eta) to ensure the tree splits map the rules without overfitting.

dtrain <- xgb.DMatrix(data = X_train, label = y_train)

params <- list(
  objective        = "binary:logistic",
  eval_metric      = "logloss",
  # Slow learning rate (0.01) avoids overfitting, compensated by high nrounds (2000)
  # to capture subtle tracking signals incrementally.
  eta              = 0.01,
  # Depth of 6 provides enough runway to capture complex/multi-layered
  # feature interactions without memorizing random tracking noise.
  max_depth        = 6,
  subsample        = 0.8,
  # Feature subsampling (0.4) blinds trees to 60% of columns, forcing diversity and
  # preventing our dominant Shot Quality variable from monopolizing the entire ensemble.
  colsample_bytree = 0.4
)

set.seed(42)
model <- xgb.train(
  params  = params,
  data    = dtrain,
  nrounds = 2000,
  verbose = 0
)


# ── 10c: PROBABILITY INFERENCE & SUBMISSION WRITE ────────────────────────────

dtest       <- xgb.DMatrix(data = X_test)
predictions <- predict(model, dtest)

submission <- data.frame(
  shot_id   = test$shot_id,
  make_prob = round(predictions, 6)
)

write.csv(submission, "submission.csv", row.names = FALSE)
cat("submission.csv written —", nrow(submission), "rows, range [",
    round(min(predictions), 3), ",", round(max(predictions), 3), "]\n")

