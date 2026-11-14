# Mã survey được chép nguyên trạng (vendored)

Không sửa tệp nào trong thư mục này. Cập nhật bằng `python scripts/vendor_survey.py --survey <clone> --commit <hash>`
rồi commit riêng `vendor: update to <hash>`. `tests/test_vendor_integrity.py` đối chiếu SHA-256 của mọi tệp với bảng dưới.

| Mục | Giá trị |
|---|---|
| Kho nguồn | `datachain-uit/RNA-Seq-FS-Survey` |
| Nhánh | `dmkd-revision` |
| Commit | `c1828d2ac9622232c1ee22f69fab7fdf2bae5a7d` |
| Ngày commit nguồn | `2026-11-13T17:22:21+07:00` |
| Ngày chép | `2026-11-14` |
| Giấy phép | MIT (`vendor/LICENSE`) |
| reference/ | kết quả GA, BPSO của survey tại commit `9b2d6a5`, lấy từ `results/survey_v2/E1_wrapper_feature_selection/` của commit nguồn |

## Tệp và SHA-256

| Tệp | SHA-256 |
|---|---|
| `LICENSE` | `4cbb12760dd0fe546bcf5ff8d595ae0900afb836cb444ed3c64332d02467be84` |
| `pytest.ini` | `95daf8c68748c8f7f06c65cad3c861d471edf1744e6ce7203d6a0a96e9485d6e` |
| `reference/E1_wrapper_feature_selection/gse10245/config.json` | `1b0355471e8370db52d7f595fc33928fdc5d68dfd6cc88d43a262290efac7fc5` |
| `reference/E1_wrapper_feature_selection/gse10245/selected_genes.csv` | `26af0bf6dae4b010b8dc75d7114e8c5c0507af31e80ef89804da269623100eca` |
| `reference/E1_wrapper_feature_selection/gse10245/selection_per_fold.csv` | `8557390d1e386340da6e05be9cb6d2dbc191d68beb3c42c8f7ab16dc29cf8664` |
| `reference/E1_wrapper_feature_selection/gse30219/config.json` | `5b274718594472405c81b2f59b69ca70ad12b4c8013a040e4d678bfeb8534d9e` |
| `reference/E1_wrapper_feature_selection/gse30219/selected_genes.csv` | `4ca15b016b52c4c4d9e3285e6302ee5c8064b3fc53879fb41f48e73abf39f7fc` |
| `reference/E1_wrapper_feature_selection/gse30219/selection_per_fold.csv` | `7d3067758c5e2e910c53d69228845c0e75e154879728d8eec6071be4cd80105e` |
| `reference/E1_wrapper_feature_selection/gse37745/config.json` | `37ab97cda7cc461f3b1bdf048bc205bb1b1d02fea1d94a9d5abaf443a4d06f04` |
| `reference/E1_wrapper_feature_selection/gse37745/selected_genes.csv` | `b6dcd1348b4d4790be64cfd2dfe82bc66f4e9d05327f2f78058fe4065f06c569` |
| `reference/E1_wrapper_feature_selection/gse37745/selection_per_fold.csv` | `246e7e66dfdbe6214658b27227331980cf5c4058ad672ce409095e4ce98a2532` |
| `requirements-kaggle.txt` | `56863d684198670e8816511b9d5deb8e105769465d87a713398390b68c0f53b8` |
| `scripts/provenance.py` | `626c0cd43e66468567c8e003a0b1d7ab61993537198463cdc7f8bcfa2ad21738` |
| `scripts/run_e1_classification.py` | `2ae6e1914aa34062fe9a841d4abd98b152131ceb31284c660cf8c11abedb05c6` |
| `scripts/run_e1_feature_selection.py` | `fa2aeb4f7c334d22e13a0ceb90f425d000e71aa0ef2d94759b058390e9c27ff3` |
| `survey_v2/__init__.py` | `231bf43695953a2481fc3cb4a83e4915aebc99270ada11cd105f3478d78d3f87` |
| `survey_v2/b7_egfr.py` | `0776f313355331aa1a0e1231278bece8a041bfa2af455bd94b925345b2b38469` |
| `survey_v2/classify.py` | `e1ee80dfc4a74052fb5ce15a2995f7c51d1f4850f98dd6c49a8bcb44a8b7b558` |
| `survey_v2/config.py` | `fa1bb8cd65c54ad3f2fd060dc66a3ad3bfa6a6ad04c985aa753508ceec94ecfa` |
| `survey_v2/data.py` | `2ddd06d005737b4a86796268c26a83b1a01a4e213a36f686bde95dd592e85be1` |
| `survey_v2/e1_classification.py` | `b9f554e04df937efcc77a302e652e5d7459947ec3f93db268395b1923411dd68` |
| `survey_v2/e1_splits.py` | `5e96b4b1e6b11e1c8d88f4f1c895818bc98682658d32d4cc42d8dcfc2a777b1f` |
| `survey_v2/fewshot.py` | `0b0c91e31ba68c7082595d06f925ef62c42a588c6089d48b15bb0e805b4e19a1` |
| `survey_v2/kstar.py` | `631dfb1a48a8eb0f7a53e79dbb48f2bc12f6a955ea97608356186532d0d9b0b4` |
| `survey_v2/methods.py` | `6a8ba14f7cdcb52d1474f9a70b6a43065a42ae53f83c6eebcdc6db67d8afe45f` |
| `survey_v2/mrmr.py` | `163252d1362a98553ee63e894bea985edf6a8672a57694c9ae37fb52b83511ae` |
| `survey_v2/panels.py` | `e3c1a534eadbb5e2907a0a5683512df47b61ac29c28f0fd4fa076b8f4566cf27` |
| `survey_v2/panels/published_panels.csv` | `7c1646b9242093dc2414f6c851efc28a0ef8694efe417b713d0803f77f4478d8` |
| `survey_v2/runlog.py` | `b2a1042ad80d5733edcd6dc376dcc4a4d27be63e414dd588c863b5a2ae955271` |
| `survey_v2/stability.py` | `10903369074a260173030df4c02400df9626a9d788a13846be39ae94bb9d1b81` |
| `survey_v2/stats.py` | `b0874e633c7b2d0c3382d2e82013b0d43096923f30ade628afdf002b19f7cf1f` |
| `survey_v2/synthetic.py` | `c19c75fa4dfad3338f04bd8f07dbd685447200d16380cd74f7af9b325864d06b` |
| `survey_v2/timing.py` | `17360e05874e668cc3b66ca92580620c109809fd99c68f3032b985286c33bc78` |
| `survey_v2/timing_child.py` | `ab236df8f597afe5b349e87c013a272811a1b767db9972dfff9280cbd79b7579` |
| `survey_v2/transfer.py` | `04e9decd8ce8270a69673f26e3174c8c77d51f8e09ae3512e19328515cf41c88` |
| `survey_v2/wrappers.py` | `f7dd05dbf9b31c35515ecbc7701284c683a0cdbfab769407ca032cc99b083f80` |
| `tests/test_e1_classification.py` | `f810f1c4881cc01fb3d3cb004aacd0eba8eb7ec568bc4004bf675ae3e45c33cc` |
| `tests/test_e1_precomputed_splits.py` | `040903b16e18556c4dfadee6d5fae68d00b767bb40b1478b25da622ef625ddb3` |
| `tests/test_provenance.py` | `83c80beb7432b6703bdf24ad143f1238b32f1c12a9507770c3d97a6ea5305061` |
| `tests/test_published_panels.py` | `819ffeaeea76fa12affb5031c79082d184af94d37dd07179eaca1fefa164026f` |
| `tests/test_runlog.py` | `9645f469f333e10977658390176c30875ad49f251fd595235082cbd6028134af` |
| `tests/test_survey_v2_methods.py` | `c686e07e530769ce3e89dcb98ce59280cbb57010ab1604f0f769e316f2aa6480` |
| `tests/test_survey_v2_stats_stability.py` | `2d62c0334be5518b1d1fa0bcf84ac9ce32b727f6a510a56f02c9d9fd94c2ffc8` |
| `tests/test_survey_v3_units.py` | `ab5c8101a295640ac4e8325ac65769b3ce69cad91c82185b58c30d7457e17b17` |
| `tests/test_v317_csv_alignment.py` | `61dd9433872701868639388b72ebebdd78156c85a3481496a11a1214603d59fb` |
| `tests/test_v3_data_layout.py` | `09d5a8d93221a55622900d7b1ef03152314ab73814202b88919537284c3ded9c` |
