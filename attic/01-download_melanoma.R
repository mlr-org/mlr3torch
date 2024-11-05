library(here)

training_jpeg_images_url = "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_JPEG.zip"
training_metadata_url = "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_GroundTruth.csv"
training_metadata_v2_url = "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_GroundTruth_v2.csv"
training_duplicate_image_list_url = "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_Duplicates.csv"

test_jpeg_images_url = "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Test_JPEG.zip"
test_metadata_url = "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Test_Metadata.csv"


urls = c(
  training_jpeg_images_url,
  training_metadata_url, training_metadata_v2_url, training_duplicate_image_list_url,
  test_jpeg_images_url,
  test_metadata_url
)

cache_dir = here("cache")
download_melanoma_file = function(url) {
  op = options(timeout = 36000) 
  on.exit(options(op))
  
  download.file(url, here(cache_dir, basename(url)))
}

mlr3misc::walk(urls, download_melanoma_file)

unzip(here(cache_dir, basename(training_jpeg_images_url)))
unzip(here(cache_dir, basename(test_jpeg_images_url)))
