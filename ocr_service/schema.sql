-- Lab results saved from /ocr?lab=true. Created by the service on first use
-- (CREATE TABLE IF NOT EXISTS), or run by hand:
--   mysql -u root ocr < ocr_service/schema.sql
-- db.py splits this file on ";" -- keep semicolons out of comments and strings.

-- One uploaded report, results and all. patient_code + sample_no identify it,
-- so saving the same report twice is refused unless it is explicitly replaced.
-- The results themselves live in the `results` JSON column exactly as
-- /ocr?lab=true returned them, so what is read back is byte-for-byte the
-- format the API produced -- no column mapping in between.
CREATE TABLE IF NOT EXISTS lab_reports (
  id                 BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
  patient_code       VARCHAR(64)  NULL COMMENT 'e.g. KCM-260910054323 (name and address are not stored)',
  sample_no          VARCHAR(64)  NULL COMMENT 'e.g. 0007-10092026',
  collected_at       DATETIME     NULL,
  received_at        DATETIME     NULL,
  source_filename    VARCHAR(255) NULL,
  engine             VARCHAR(16)  NOT NULL COMMENT 'surya | tesseract',
  results_count      INT UNSIGNED NOT NULL,
  needs_review_count INT UNSIGNED NOT NULL COMMENT 'results a person should check against the document',
  results            JSON         NOT NULL COMMENT 'lab.results: [{name, percent?, value, flag, unit, ref_range, category}]',
  review             JSON         NULL COMMENT 'lab.review: review[i] describes results[i]',
  created_at         DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uq_lab_reports_patient_sample (patient_code, sample_no),
  KEY ix_lab_reports_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
