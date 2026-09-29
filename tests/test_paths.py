from annanas import paths


def test_processed_dirs_live_under_datasets():
    assert paths.PROCESSED_DIR == paths.DATASETS_DIR / "processed"
    assert paths.EVENTS_DIR == paths.PROCESSED_DIR / "events"
    assert paths.EDA_DIR == paths.PROCESSED_DIR / "eda"
