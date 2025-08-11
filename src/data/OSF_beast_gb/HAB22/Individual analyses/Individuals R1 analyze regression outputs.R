library(broom)
library(dplyr)
library(ggplot2)
library(tidyr)
library(tibble)
library(brms)

setwd("C:/Users/plonsky/Dropbox/CPC 2018/NHB23 files/R1 files/Final files for upload/HAB22/Individual analyses")

########### analyze mixed effects models

# load data
load("Bayesian regressions outputs.RData")

# Extract fixed effects from each model and store them in a data frame.
fixed_df <- lapply(seq_along(cv_models), function(i) {
  m <- cv_models[[i]]
  fe <- fixef(m)
  fe_df <- as.data.frame(fe)         # columns: Estimate, Est.Error, Q2.5, Q97.5
  fe_df$term <- rownames(fe)          # add a column for predictor names
  fe_df$fold <- i                     # record the CV fold number
  fe_df
}) %>% bind_rows()
fixed_summary <- fixed_df %>%
  group_by(term) %>%
  summarize(
    mean_est = mean(Estimate, na.rm = TRUE),
    sd_est = sd(Estimate, na.rm = TRUE),
    mean_lower = mean(Q2.5, na.rm = TRUE),
    mean_upper = mean(Q97.5, na.rm = TRUE),
    n_folds = n(),
    .groups = "drop"
  )

# Define custom labels for facet titles
custom_labels <- c(
  "diffEV" = "EV_difference",
  "IBEAST_GB_predM0.5" = "BEAST-GB (centered)",
  "pBbet_Unbiased1" = "Unbiased sampling",
  "Intercept" = "intercept",
  "pBbet_Uniform" = "Uniform sampling",
  "pBbet_Sign1" = "Sign sampling",
  "diffMins" = "Pessimism (diff of mins)"
)

# Define the custom order of terms
custom_order <- c("IBEAST_GB_predM0.5", "diffEV", "pBbet_Unbiased1",
                  "pBbet_Uniform", "pBbet_Sign1", "diffMins","Intercept")  # Adjust the order as needed

# Convert 'term' into a factor with the specified order
fixed_df$term <- factor(fixed_df$term, levels = custom_order)

ggplot(fixed_df, aes(x = factor(fold), y = Estimate)) +
  geom_pointrange(
    aes(ymin = Q2.5, ymax = Q97.5),
    color = "blue",
    size = 0.5,
    fatten = 2
  ) +
  facet_wrap(~ term, scales = "free_y", labeller = labeller(term = custom_labels)) +
  labs(
    x = "Fold",
    y = "Fixed Effect (with 95% CI)"
  ) +
  theme_minimal() +
  theme(strip.text.x = element_text(size= 12),
        axis.text.y = element_text(size=11))

### random effects
random_effects_list <- lapply(seq_along(cv_models), function(i) {
  model <- cv_models[[i]]
  # Extract the posterior mean estimates (the "Estimate" component)
  re_array <- ranef(model)$subj_id  # array with dimensions: subjects x parameters x statistic
  re_mat <- re_array[,  "Estimate",]  # matrix: rows = subjects, columns = parameters
  # Convert the matrix to a data frame and add subject IDs as a column.
  df <- as.data.frame(re_mat)
  df <- rownames_to_column(df, var = "subj_id")
  df$fold <- i
  return(df)
})

# Combine the data from all folds.
random_effects_df <- bind_rows(random_effects_list)

# Now reshape to long format: one row per subject per predictor.
random_effects_tidy <- random_effects_df %>%
  pivot_longer(
    cols = -c(subj_id, fold),
    names_to = "predictor",
    values_to = "estimate"
  )

# Summarize the estimates for each predictor across all folds.
random_summary <- random_effects_tidy %>%
  group_by(predictor) %>%
  summarize(
    mean_est = mean(estimate, na.rm = TRUE),
    sd_est = sd(estimate, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  )

# Aggregate random effects for each subject and predictor across all folds.
random_effects_avg <- random_effects_tidy %>%
  group_by(subj_id, predictor) %>%
  summarize(
    avg_estimate = mean(estimate, na.rm = TRUE),
    sd_estimate  = sd(estimate, na.rm = TRUE),
    n_folds      = n(),
    .groups = "drop"
  )
# Visualize these subject-level average random effects with violin plots.
random_effects_avg$predictor <- factor(random_effects_avg$predictor, 
                                       levels = custom_order)
ggplot(random_effects_avg, aes(x = predictor, y = avg_estimate)) +
  geom_violin(
    fill = "skyblue", alpha = 0.6, trim = FALSE,
    scale = "width",      # each violin has same maximum width
    adjust = 1.2          # enlarge bandwidth to avoid overly skinny violins
  ) +
  geom_boxplot(
    width = 0.15, outlier.shape = NA
  ) +
  labs(
    x = "Predictor",  y = "Average Random Effect (across folds)") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 12),
    axis.text.y = element_text(size=11))+
  scale_x_discrete(
    breaks = custom_order,  # Order of the x-axis categories
    labels = custom_labels[custom_order])  # Corresponding labels from custom_labels

# Pivot random_effects_df to long format.
random_long <- random_effects_df %>%
  pivot_longer(
    cols = -c(subj_id, fold),
    names_to = "term",
    values_to = "random_effect"
  )

# Join the fixed effects with the random effects by fold and term.
combined_effects <- random_long %>%
  left_join(fixed_df %>% select(term, Estimate, fold), by = c("term", "fold")) %>%
  mutate(total_effect = Estimate + random_effect)

# For each subject and each predictor (term), average the total_effect across folds.
subject_sensitivity <- combined_effects %>%
  group_by(subj_id, term) %>%
  summarize(
    avg_total_effect = mean(total_effect, na.rm = TRUE),
    sd_total_effect = sd(total_effect, na.rm = TRUE),
    n_folds = n(),
    .groups = "drop"
  )
subject_sensitivity_wide <- subject_sensitivity %>%
  pivot_wider(
    names_from = term,
    values_from = c(avg_total_effect, sd_total_effect),
    names_glue = "{term}_{.value}"
  )

summary(subject_sensitivity_wide)

# Proportion of subjects whose average total effect exceeds 0
prop_positive <- subject_sensitivity %>%
  group_by(term) %>%
  summarize(
    n_subj = n(),
    n_pos  = sum(avg_total_effect > 0, na.rm = TRUE),
    prop_pos = n_pos / n_subj,
    .groups = "drop"
  )

# Accounting for variability: "consistently positive" if 
# avg_total_effect - 2*sd_total_effect > 0
consistency <- subject_sensitivity %>%
  mutate(
    consistently_pos = (avg_total_effect - 2 * sd_total_effect > 0),
    consistently_neg = (avg_total_effect + 2 * sd_total_effect < 0)
  ) %>%
  group_by(term) %>%
  summarize(
    n_subj = n(),
    prop_avg_pos = mean(avg_total_effect > 0),
    prop_consistently_pos = mean(consistently_pos, na.rm = TRUE),
    prop_consistently_neg = mean(consistently_neg, na.rm = TRUE),
    mean_sd = mean(sd_total_effect, na.rm = TRUE),
    median_sd = median(sd_total_effect, na.rm = TRUE),
    .groups = "drop"
  )
consistency

random_effects_context <- random_effects_avg %>%
  mutate(
    context = sub("_[^_]*$", "", subj_id),
    subject = sub(".*_", "", subj_id)
  )
random_effects_context$predictor <- factor(random_effects_context$predictor, 
                                           levels = custom_order)
ggplot(random_effects_context, aes(x = context, y = avg_estimate)) +
  geom_boxplot(outlier.shape = NA, fill = "lightblue") +
  facet_wrap(
    ~ predictor, 
    scales = "free_y", 
    labeller = labeller(predictor = as_labeller(custom_labels))
  ) + 
  labs(
    x = "Context",
    y = "Average Random Slope"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size=10),
        strip.text.x = element_text(size=11))

# Check differences statistically:
slope_data <- filter(random_effects_context, predictor == "pBbet_Unbiased1")
kruskal.test(avg_estimate ~ context, data = slope_data)

slope_data <- filter(random_effects_context, predictor == "diffEV")
kruskal.test(avg_estimate ~ context, data = slope_data)
slope_data <- filter(random_effects_context, predictor == "pBbet_Sign1")
kruskal.test(avg_estimate ~ context, data = slope_data)

