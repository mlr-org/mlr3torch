training_jpeg_images_url = "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_JPEG.zip"
training_metadata_url = "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_GroundTruth.csv"
training_metadata_v2_url = "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_GroundTruth_v2.csv"
training_duplicate_image_list_url = "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_Duplicates.csv"

test_jpeg_images_url = "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Test_JPEG.zip"
test_metadata_url = "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Test_Metadata.csv"


urls = c(
  training_jpeg_images_url
  # training_metadata_url, training_metadata_v2_url, training_duplicate_image_list_url
)

unzip(here(cache_dir, basename(training_jpeg_images_url)))

options(timeout = 36000) 

download_melanoma_file = function(url) {
  download.file(url, here::here("cache", basename(url)))
}

mlr3misc::walk(urls, download_melanoma_file)
