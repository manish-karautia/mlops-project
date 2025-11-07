terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = "manish07"
  region  = "us-central1"
}

resource "google_storage_bucket" "mlops_bucket" {
  name                     = "mlops-githubdemo-bucket"
  location                 = "US"
  force_destroy            = true
  public_access_prevention = "enforced"
}
