# Setting HTTP User Agent to identify OS, such that P3M can detect compatibility
options(HTTPUserAgent = sprintf("R/%s R (%s)", getRversion(), paste(getRversion(), R.version["platform"], R.version["arch"], R.version["os"])))

# Ensure curl is used for downloading packages
options(download.file.method = "curl")

# Enable verbose output for curl and again set HHTP user agent
options(download.file.extra = paste(
  # Follow redirects, show errors, and display the HTTP status and URL
  '-fsSL -w "%{stderr}curl: HTTP %{http_code} %{url_effective}\n"',
  # Configure the R user agent header to install Linux binary packages
  sprintf('--header "User-Agent: R (%s)"', paste(getRversion(), R.version["platform"], R.version["arch"], R.version["os"]))
))

# for ubuntu:
options(
  repos = c(CRAN = "https://packagemanager.posit.co/cran/__linux__/jammy/latest")
)
