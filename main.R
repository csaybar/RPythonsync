# Create a dataset for deep learning using r-spatial
library(rgeeExtra)
library(spatstat)
library(mapview)
library(stars)
library(mmap)
library(rgee)
library(sf)

source("utils.R")

# Gee with R
ee_Initialize() # Initialize gee

# 1. Define your study area
roi <- st_read(system.file("shp/arequipa.shp", package="rgee")) %>%
  st_as_sfc() %>%
  st_transform("EPSG:32718")
mapview(roi)

# 2. Sampling points hard-core point process (HCPP)
# In simple words, it is a way of placing points in a
# given space such that no two points are closer to each other
# than a specified minimum distance. This minimum distance is
# referred to as the "hard-core distance."
beta <- 100; R <- 20*(1000)


# Gibbs point process
# beta -> interaction between points
# R -> hard-core distance
model <- rmhmodel(
  cif="hardcore", 
  par = list(beta=beta, hc=R),
  w = as.owin(roi)
) # hard-core model for generating the point pattern

# We uses random sampling and iterative steps to approximate the 
# target distribution
results <- rmh(model, start = list(n.start = 100))

points <- st_as_sf(results)[2:results$n,] %>%
  st_set_crs(32718)
# mapview(points)  | mapview(sf::st_sample(roi,results$n))

index <- 1
# 3. Download Sentinel-2 Data L1C and L2A
for (index in 1:100) {
  print(index)
  point <- points[index,]$geom
  s2_patch <- download_image(point)
  write_stars(
    obj = s2_patch,
    dsn = sprintf("data/POINT_%03d__%s.tif", index, attributes(s2_patch)$name)
  )
}

# 5. Make a prediction using a trained model
file <- "data/POINT_100__20210429T145719_20210429T150434_T18LZH.tif"
model <- torch$jit$load("model.pt")
rr_ii <- prediction(file, model)
