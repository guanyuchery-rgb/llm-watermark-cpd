# I/O adapter only: evaluate the original R numerical functions and NOT loop.
# No plotting code, package installation or algorithm replacement is performed.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 9) stop("Expected detect_dir output_dir samples tokens window minimum permutations threshold seed")
detect_dir <- args[1]
output_dir <- args[2]
sample_count <- as.integer(args[3])
token_count <- as.integer(args[4])
rolling_window_size <- as.integer(args[5])
minimum_length <- as.integer(args[6])
significance_permutation_count <- as.integer(args[7])
threshold <- as.numeric(args[8])
experiment_seed <- as.integer(args[9])
script_arg <- grep("^--file=", commandArgs(), value = TRUE)[1]
root <- dirname(dirname(normalizePath(sub("^--file=", "", script_arg))))

functions <- c("get_seeded_intervals", "ks_statistic", "permute_pvalues", "segment_significance")
found <- character()
for (expr in parse(file.path(root, "4-seedbs.R"))) {
  if (is.call(expr) && identical(expr[[1]], as.name("<-")) &&
      as.character(expr[[2]])[1] %in% functions) {
    eval(expr)
    found <- c(found, as.character(expr[[2]])[1])
  }
}
if (!setequal(found, functions)) stop("Could not locate original SeedBS functions")

# Extract the original loop text without evaluating the paper's plotting pipeline.
not_source <- readLines(file.path(root, "5-not.R"), warn = FALSE)
loop_start <- grep("while (nrow(potential_change_points) > 0)", not_source, fixed = TRUE)
loop_end <- grep("result_df <- result_df[order(result_df$change_point_index), ]", not_source, fixed = TRUE)
if (length(loop_start) != 1 || length(loop_end) != 1 || loop_end <= loop_start) {
  stop("Could not locate the original NOT selection loop")
}
not_loop <- parse(text = not_source[loop_start:(loop_end - 1)])[[1]]

intervals <- get_seeded_intervals(token_count - rolling_window_size, unique.int = TRUE)
intervals <- intervals[intervals[, 2] - intervals[, 1] >= minimum_length, , drop = FALSE]
intervals <- intervals + rolling_window_size / 2
if (nrow(intervals) == 0) stop("No seeded intervals remain")
fields <- c("sample", "metric", "interval", "from", "to", "segment_length",
            "index_within_segment", "change_point_index", "significance")
candidates <- as.data.frame(setNames(replicate(length(fields), numeric(0), simplify = FALSE), fields))
changes <- candidates
for (sample in seq_len(sample_count)) {
  matrix <- matrix(NA_real_, nrow = token_count, ncol = 2)
  for (position in seq_len(token_count)) {
    path <- file.path(detect_dir, paste0(sample - 1, "-", position - 1, ".csv"))
    row <- read.csv(path, header = FALSE)
    if (!identical(dim(row), c(1L, 2L))) stop(paste("Invalid detection shape:", path))
    matrix[position, ] <- as.numeric(unlist(row[1, ], use.names = FALSE))
  }
  for (index in seq_len(nrow(intervals))) {
    from <- intervals[index, 1]
    to <- intervals[index, 2]
    subset <- matrix[from:to, , drop = FALSE]
    if (any(!is.finite(subset)) || any(subset < 0 | subset > 1)) stop("Invalid interior p-values")
    # Original Slurm invocation reset set.seed(1) for each interval task.
    set.seed(experiment_seed)
    result <- apply(subset, 2, segment_significance)
    for (metric in seq_len(ncol(subset))) {
      candidates <- rbind(candidates, data.frame(
        sample = sample - 1, metric = metric - 1, interval = index - 1,
        from = from, to = to, segment_length = to - from,
        index_within_segment = result[1, metric],
        change_point_index = result[1, metric] + from - 1,
        significance = result[2, metric]))
    }
  }
  for (metric in 0:1) {
    potential_change_points <- candidates[
      candidates$sample == sample - 1 & candidates$metric == metric &
        candidates$significance <= threshold, , drop = FALSE]
    result_df <- potential_change_points[FALSE, ]
    eval(not_loop)
    result_df <- result_df[order(result_df$change_point_index), , drop = FALSE]
    changes <- rbind(changes, result_df)
  }
  cat("Original R segmentation sample", sample, "of", sample_count, "\n")
}
for (name in c("seedbs.csv", "changepoints.csv", "intervals.csv")) {
  if (file.exists(file.path(output_dir, name))) stop(paste("Refusing to overwrite", name))
}
write.csv(candidates, file.path(output_dir, "seedbs.csv"), row.names = FALSE)
write.csv(changes, file.path(output_dir, "changepoints.csv"), row.names = FALSE)
write.csv(data.frame(interval = seq_len(nrow(intervals)) - 1,
                     from = intervals[, 1], to = intervals[, 2]),
          file.path(output_dir, "intervals.csv"), row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(output_dir, "r-session.txt"))
