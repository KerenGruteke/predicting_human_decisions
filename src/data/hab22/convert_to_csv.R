# Set working directory to the location of the RData file
setwd("src/data/hab22")  # Adjust if needed

# Load the .RData file
load("HAB22 all data with Stew1C_Uni.RData")

# List all objects loaded
loaded_objects <- ls()
cat("Loaded objects:\n")
print(loaded_objects)

# Loop through objects and save data frames to CSV
for (obj_name in loaded_objects) {
  obj <- get(obj_name)
  if (is.data.frame(obj)) {
    out_file <- paste0(obj_name, ".csv")
    write.csv(obj, out_file, row.names = FALSE)
    cat(paste("Exported:", out_file, "\n"))
  } else {
    cat(paste("Skipped (not a data frame):", obj_name, "\n"))
  }
}
