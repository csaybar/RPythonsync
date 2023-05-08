library(reticulate)
library(stars)
library(terra)

torch <- import("torch", convert = FALSE)
np <- import("numpy", convert = FALSE)


prediction <- function(file, model) {
    rstars_file <- read_stars(file)
    l1c <- merge(split(rstars_file, "band")[c("L1C_B8", "L1C_B4", "L1C_B3", "L1C_B2")])/10000
    l1c_array <- l1c[[1]]

    # Prepare data for prediction
    S2l1c <- l1c_array
    rr <- np$moveaxis(S2l1c, -1L, 0L)
    rr <- np$expand_dims(rr, 0L)
    rr <- torch$from_numpy(rr)$float()

    yhat <- model$forward(rr)$detach()$numpy()$squeeze()
    yhat_pred <- py_to_r(np$moveaxis(yhat, 0L, -1L))
    yhat_pred <- st_as_stars(yhat_pred)
    yhat_pred <- st_set_crs(yhat_pred, st_crs(rstars_file))
    yhat_pred <- st_set_dimensions(yhat_pred, names = c("x", "y", "band"))

    # change delta
    attr(yhat_pred, "dimensions")$x$delta <- 10
    attr(yhat_pred, "dimensions")$y$delta <- -10
    attr(yhat_pred, "dimensions")$x$offset <- attr(rstars_file, "dimensions")$x$offset
    attr(yhat_pred, "dimensions")$y$offset <- attr(rstars_file, "dimensions")$y$offset

    # Select only the fourth bands    
    write_stars(l1c, "L1C.tif")
    write_stars(yhat_pred, "L2A_AI.tif")
}


download_image <- function(point) {
  # From sf to Earth Engine
  ee_point <- sf_as_ee(point)  

  # How many images are there at this point?
  ic <- ee$ImageCollection("COPERNICUS/S2_SR_HARMONIZED") %>%
    ee$ImageCollection$filterBounds(ee_point) %>%
    ee$ImageCollection$filterDate("2021-01-01", "2021-12-31") %>%
    ee$ImageCollection$filter(ee$Filter$lt('CLOUDY_PIXEL_PERCENTAGE', 5))

  # Randomly select a image
  random_image <- sample(length(ic), 1)
  s2img <- ic[[random_image]]
  s2_id <- s2img$get("system:index")$getInfo()

  # Obtain the geotransform of the image
  projection <- s2img[["B4"]]$projection()$getInfo()
  geotransform <- projection[["transform"]]
  crs <- projection[["crs"]]

  # Fix sampled point
  xy <- as.numeric(points[index,]$geom[[1]])
  new_x <- geotransform[3] + round(xy[1] - geotransform[3]) + 10/2
  new_y <- geotransform[6] + round(xy[2] - geotransform[6]) + 10/2

  # Merge band - L1C and L2A
  S2_BANDS <- c("B8", "B4", "B3", "B2")

  s2_l1c <- sprintf("COPERNICUS/S2_HARMONIZED/%s", s2_id) %>%
    ee$Image() %>%
    '[['(S2_BANDS) %>%
    ee$Image$rename(sprintf("L1C_%s", S2_BANDS))

  s2_l2a <- sprintf("COPERNICUS/S2_SR_HARMONIZED/%s", s2_id) %>%
    ee$Image() %>%
    '[['(S2_BANDS) %>%
    ee$Image$rename(sprintf("L2A_%s", S2_BANDS))
  s2_total <- s2_l1c$addBands(s2_l2a)

  # From EarthEngine to stars
  s2_patch <- ee_as_rast(
    image = s2_total,
    crs_kernel = crs,
    ee_point = ee_point
  ) %>% merge()
  attributes(s2_patch)$name <- s2_id
  s2_patch
}



ee_as_rast <- function(image, crs_kernel, ee_point) {
  # 4.7 Create a 511x511 tile (list -> data.frame -> stars)

  band_names_s2 <- image$bandNames()$getInfo()
  band_names <- c(band_names_s2, "x", "y")

  # Obtain information in a rectangle kernel
  # From GEE to local (EEimage -> JSON)
  s2_img_array <- image %>%
    ee$Image$addBands(ee$Image$pixelCoordinates(projection = crs_kernel)) %>%
    ee$Image$neighborhoodToArray(
      kernel = ee$Kernel$rectangle(255, 255, "pixels")
    ) %>%
    ee$Image$sampleRegions(
      collection = ee_point,
      projection = crs_kernel,
      scale = 10) %>%
    ee$FeatureCollection$getInfo()


  # From GeoJSON to data.frame
  extract_fn <- function(x) {
    as.numeric(unlist(s2_img_array$features[[1]]$properties[x]))
  }
  image_as_df <- do.call(cbind,lapply(band_names, extract_fn))
  colnames(image_as_df) <- band_names
  image_as_df <- as.data.frame(image_as_df)

  # From data.frame to stars
  as_stars <- lapply(
    X = band_names_s2,
    FUN = function(z) st_as_stars(image_as_df[c("x", "y", z)])
  ) %>% do.call(c, .)
  st_crs(as_stars) <- crs_kernel
  as_stars
}

