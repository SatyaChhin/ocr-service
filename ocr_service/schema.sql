-- Lab results saved from /ocr?lab=true. Created by the service on first use
-- (CREATE TABLE IF NOT EXISTS), or run by hand:
--   mysql -u root ocr < ocr_service/schema.sql
-- db.py splits this file on ";" -- keep semicolons out of comments and strings.

-- One uploaded report. patient_code + sample_no identify it, so saving the
-- same report twice is refused unless it is explicitly replaced.
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
  created_at         DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uq_lab_reports_patient_sample (patient_code, sample_no),
  KEY ix_lab_reports_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- One row per test result. The first columns mirror the JSON format:
-- test_name, percent, value, flag, unit, ref_range, section.
CREATE TABLE IF NOT EXISTS lab_results (
  id             BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
  report_id      BIGINT UNSIGNED NOT NULL,
  position       SMALLINT UNSIGNED NOT NULL COMMENT 'order on the report',
  test_name      VARCHAR(128) NOT NULL,
  percent        DECIMAL(10,4) NULL COMMENT 'differential rows only',
  value          VARCHAR(255) NOT NULL COMMENT 'as reported: 9.9, 744, O Rh (D): Positive, < 0.5',
  value_numeric  DECIMAL(18,6) NULL COMMENT 'value when it is a plain number, for queries',
  flag           VARCHAR(4)   NULL COMMENT 'H | L | HH | LL | *',
  unit           VARCHAR(32)  NULL,
  ref_range      VARCHAR(64)  NULL,
  section        VARCHAR(128) NULL,
  needs_review   TINYINT(1)   NOT NULL DEFAULT 0,
  review_notes   TEXT         NULL COMMENT 'JSON list of {code, params}',
  ocr_confidence DECIMAL(5,2) NULL,
  CONSTRAINT fk_lab_results_report FOREIGN KEY (report_id) REFERENCES lab_reports (id) ON DELETE CASCADE,
  KEY ix_lab_results_test (test_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
