"""Frozen, solver-free contract for the H23 qualification diagnostic.

This module deliberately contains data only.  Both the production diagnostic
runner and its independent verifier import these pins, while the verifier never
imports either runner module or Gurobi.
"""

from __future__ import annotations


H23_DIAGNOSTIC_PLAN_ID = "h23-qualification-diagnostic-v2-60s-geometry30"
H23_DIAGNOSTIC_PLAN_SCHEMA = "mlnd-h23-qualification-diagnostic-plan-v2"
H23_DIAGNOSTIC_RUN_SCHEMA = "mlnd-h23-qualification-diagnostic-run-v2"
H23_DIAGNOSTIC_CHECKPOINT_SCHEMA = (
    "mlnd-h23-qualification-diagnostic-checkpoint-v2"
)
H23_DIAGNOSTIC_SUMMARY_SCHEMA = "mlnd-h23-qualification-diagnostic-summary-v2"
H23_DIAGNOSTIC_MAX_JOBS = 24

FROZEN_BASE_COMMIT = "47efb94de7ac6c1531eca98709f9ac0e2b1c47d4"
FROZEN_MANIFEST_SHA256 = (
    "6f89250a2f2d3b2c3ec23593e985e1ca3ce018db96a4ed3febf651f19290a8e8"
)
FROZEN_MANIFEST_FILE_SHA256 = (
    "3b8d71f6018a50431a0b3d1fad7d18c71c041b12c8787610ab7b6304d67e4959"
)
FROZEN_SOURCE_RUN_ID = (
    "development-calibration-run-"
    "ea3d709c838db11ea0b0704d49d81198a6350fc0b89deea46175f7c42e26e6eb"
)
FROZEN_SOURCE_RUN_IDENTITY_SHA256 = (
    "ea3d709c838db11ea0b0704d49d81198a6350fc0b89deea46175f7c42e26e6eb"
)
FROZEN_SOURCE_RUN_IDENTITY_FILE_SHA256 = (
    "e5b1fe84afc2a235a59677ade2544d7b9071ad9b8a0b36df145a6ecdaa26b997"
)
FROZEN_SOURCE_SUMMARY_SHA256 = (
    "114f47eba33417e743e208e998abfedf91c38b3e2111b54d93f131d9dd4154f3"
)
FROZEN_SOURCE_SUMMARY_FILE_SHA256 = (
    "ca368d4402f85922f971ed2958dd62afc525d0c749785de9213ebccfcb07ca78"
)
FROZEN_SOURCE_IMPLEMENTATION_SHA256 = (
    "cae03efb0355eb72b6d4dedcfcffd80d92592397a20ff522929760812423038f"
)

REGISTERED_POLICY_PAYLOAD = {
    "optimality_tolerance": 1e-6,
    "minimum_savings_fraction": 1e-3,
    "minimum_service_fraction": 1e-3,
    "minimum_h_open_fraction": 0.15,
    "maximum_h_open_fraction": 0.85,
    "large_scale_open_h_target": [50, 60],
}

REGISTERED_NATIVE_SETTINGS_PAYLOAD = {
    "time_limit_seconds": 60.0,
    "mip_gap": 0.01,
    "solver_seed": 42,
    "threads": 1,
    "output_flag": 0,
    "method": -1,
    "miqcp_method": -1,
    "mip_focus": 0,
    "heuristics": 0.05,
    "no_rel_heur_time": 0.0,
    "presolve": -1,
    "native_cuts": -1,
    "root_cut_passes": 2,
    "numeric_focus": 0,
    "feasibility_tolerance": 1e-6,
    "optimality_tolerance": 1e-6,
    "integer_feasibility_tolerance": 1e-5,
}

REGISTERED_GEOMETRY_POLICY_PAYLOAD = {
    "enabled": True,
    "per_attempt_seconds": 30.0,
    "start_node_limit": 500,
    "plan_version": "qualification-geometry-start-v2-system-energy",
    "active_tolerance_multiplier": 2.0,
    "required_service_multiplier": 1.05,
    "small_support_schedule": [12, 8, 15],
    "medium_support_schedule": [26, 17, 34],
    "large_support_schedule": [55, 50, 60],
}

LEGACY_IMPLEMENTATION_FILES = (
    "src/config.py",
    "src/data_processor/loader.py",
    "src/data_processor/instance.py",
    "src/model/gurobi_model.py",
    "src/benchmark/branch_cut/artifacts.py",
    "src/benchmark/branch_cut/hashing.py",
    "src/benchmark/branch_cut/case_builder.py",
    "src/benchmark/branch_cut/calibration_builder.py",
    "src/benchmark/branch_cut/calibration_runner.py",
    "src/benchmark/branch_cut/calibration_verifier.py",
    "src/benchmark/branch_cut/qualification.py",
    "src/benchmark/branch_cut/qualification_geometry_start.py",
    "src/benchmark/branch_cut/qualification_runner.py",
    "src/benchmark/branch_cut/resource_lock.py",
    "src/benchmark/branch_cut/h23_qualification_diagnostic_contract.py",
    "src/benchmark/branch_cut/h23_qualification_diagnostic.py",
    "src/benchmark/branch_cut/h23_qualification_diagnostic_verifier.py",
)
IMPLEMENTATION_FILES = (
    *LEGACY_IMPLEMENTATION_FILES,
    "src/benchmark/branch_cut/qualification_incumbent_reuse.py",
    "src/benchmark/branch_cut/h23_incumbent_reuse_contract.py",
)

# Ordered exactly as the 24 small/seed-stage-0 jobs in the frozen manifest.
# Each row is (source job id, canonical filename, file SHA256, checkpoint self-hash).
FROZEN_SOURCE_CHECKPOINTS = (
    (
        "development-calibration-job-d6787961a8836ef4fa37bdcc91558fa8ad8ec54fc8cfb5bc3b26210fd3527ea6",
        "job-000-7697cdcc8f216707514596a2585834979c3611d31a8cdaaca1b70397ec7d08ea.json",
        "504bbc21f3a534022927a6bca77bfb3b26571e32b6a5fb24a15b68d0ee7fdb3f",
        "e16c28b4abb996f15e260af1cbdec4f269f995211109857c66ec641cb555d138",
    ),
    (
        "development-calibration-job-d2dc099f721247d26fd3426dec1034b95fa1ef6c0be05db19986d6bfb3e273d9",
        "job-001-6830a03ec75145afb5c26528a06f3190e4f4fb43f1425ede4c44ef7c27a8fe22.json",
        "e49cad302df34210c1215da91eb02adc035ad6a63d71de0d1fec051d97a124c4",
        "5565c42cd7840c41d4a4c21dc24f673fe9e872f1e61626b8ac195468bb13bb29",
    ),
    (
        "development-calibration-job-e0189f6e741b872f3336a4ed267bba89b398b837feb73aa37d2f3b81e534c42d",
        "job-002-b5c170109b78788cadd616b4db03fd895f3c002fa925e1031af69e5759353207.json",
        "d5b9fa359a3aa7a9addd78de568dbadf489d43b84d58650ac0a2680ce5916ba8",
        "36740d3884b81e4a161bba80983f07c65e345cb2f2ca28a3be6a01de4ec2e32c",
    ),
    (
        "development-calibration-job-36dc31f8dd360d1cda89f16f80ecf7344d661b10d5772e2293b1b1828a218c37",
        "job-003-7b1f550ff036b5f8225140c3e9a265d1ae8fca9bea985ca5ec93765694fde360.json",
        "24a8efe2b30e961d9594aebfa6cdde1f32569a366abc64dfdc7770584275c5d0",
        "be2f56c946d1f0c48ded06128d676fd20110d57c71a020383ba6b34e31c9668e",
    ),
    (
        "development-calibration-job-e774fc72a1c74f63c16239978a3f7ce1871b8d0f56ad723cc549b6babe1f617e",
        "job-004-065e52cdd0ed5220600eb4ac0ce1ca117034a8634168e585abe27194e512c7ad.json",
        "f2022b7a963ddee217778b2bd6a1c5aadbdacbcfe213117fe595de302991817f",
        "5f5a34ae604af05f30f33e8b9fc387ff6112bd45768c330e41141a54be2d5632",
    ),
    (
        "development-calibration-job-efe9525cefe6322bd82dc0acb63819a5dfca59501292119e2f5f99dcd649bff6",
        "job-005-560229dd2a674e446f64a189641457479e7048b3fd70758c8592bfd30f3e015e.json",
        "d3a1133343ee1e531185e0f71cd08fcf1f9d266187a9a45d87b0a93d3f4a80e0",
        "4ecb6e2c5d44d70b1de0b330e3d5218163f84bf9b52acd888bf8eb400662280e",
    ),
    (
        "development-calibration-job-69df8db17be27ab603d52b644bc5994de0912b7d5cca6e4221474dd235b4b203",
        "job-006-5fd4dac775f2c5ce4a37888334d0d26804d5671a5421d89642b707dcb668aba6.json",
        "6cefbb62b448be3d103e4a1c642a2b1bab00bb0ea60a616a484b1ee4bbd93bee",
        "3e0af9cf6dee2543a387524a7ceb3ad94687b76895b05c3ce65b92b604d94ba2",
    ),
    (
        "development-calibration-job-b12edddb0ce854f03496147547c072317489d9bf67441b693a12a21ac819e94f",
        "job-007-22c2e794259e492a0fb5de4bb81bcb10ee788fec53c1181f0fc179589517c43f.json",
        "ea10ec5ceb16733fadc071ff35706b3df2acb1dda3898d0e061ce328ffc018d7",
        "c4a96ab64362211fc5cc26b2a93f9791cde3e72951babde26c6f32ed06c376c3",
    ),
    (
        "development-calibration-job-50196c70fd60b261cc7559f185739cbad10f16a93bac044d1a0c84daef23010a",
        "job-008-d0182add7a578209cc729a7188e78dd727c4fef3b29ec1c7c9fa3775558c56e9.json",
        "bd8e0630aa8bb25de80e3aaa016b1833ece3b061812df10bd5952410108e5d51",
        "1ee06d0e0fb482bb1a6d47171db8c0c6b7cb5cee28d261e5ad4cbfad8eac25e0",
    ),
    (
        "development-calibration-job-0edc37b9458d149602d1d34eea4af9933c37b076fe4d1fe2076c9aa72376f378",
        "job-009-3bb13aaff37c968143fb11abe660204c91a4e86b2904745075356abfbb2aee24.json",
        "7ddf95d6fef26ca0ff456b72b16e6b3d3cca4e13d5010a9c40782cbe3d9e221f",
        "e68b1217745ec12310afeb270be650b9db820f54b9b5b2bf8b41abc89a39b4c8",
    ),
    (
        "development-calibration-job-139a3215714636f322328ba6a919df623eb68cc9c22e57ab2868d8deb45f647d",
        "job-010-e42d2a2b8e5bb0793845412a0d82f4498cd23f9cf7400100fb782d6a8cc2555e.json",
        "60f9540c0a3d8b447a0876856a38c71c8dd9727e53ba772424082cddfe530510",
        "cbebec95d31230ee5e3ca900937bf7cd7ea45148ab88540ba632e5a9654b7118",
    ),
    (
        "development-calibration-job-94d54720a5c959b874235ce7080e99066a1405d95a8c53293c9b0c8e27dacb39",
        "job-011-9be8fc0eae408d006813cfd6201aa3932ad6298fbf681f78ed51a11044655ba5.json",
        "3452a063777a60f0395ac15618aa21efba22ba0b52de84e81d54d22d115ccfbd",
        "8e01901e02e4b84918b2abc65765f62d1499691a19481293ae1471ad56b45ac2",
    ),
    (
        "development-calibration-job-0efb84621915a3b97b93008621a83d88f389b63f15ab5c0a1a63c5dfb3357504",
        "job-012-8629b5c2cff3476205d569cd43ae41ace8628fc59063c028f94358d8278790dd.json",
        "8eb9ccd76c79ea6d2473f8c8933c05c00265dfb69d91fcd155d41a67c96c8c92",
        "10b068305ec1fe71513d06d6f6f6cd12eadff5126e831e1df0bf26caa819b38b",
    ),
    (
        "development-calibration-job-65f66caf19289c6191d0ae58f9d74fbd9e769149aa63d03547b3a733d703ed27",
        "job-013-c63c8ace3d6bd80cf787b7379067d5096bb294973438ec55c21c5a3d0e3f4d09.json",
        "848554f37227c6398cf1cb9a4bfa1cf9ba516f1d14c2653025922ce70cf0baa8",
        "8074cef3754604785ff2476cf3502cb3be61db9137d4ff2354d0274ae059c8de",
    ),
    (
        "development-calibration-job-bb76e74fcab72a75071e7a80746f1d807aa54124c8ed02551d411336f9f5ac45",
        "job-014-d25914a1cbbb62e9b8385be00f6629a996b7ede556a9fe3c581b0c6f95c941da.json",
        "bdaed4d4b723681d64f748016287de9060f599b5de1ef1581ca00d60706c9341",
        "cacc1c26b66abe6a70de4a6eed61ec2b8952201f2c119058267c34638861bf80",
    ),
    (
        "development-calibration-job-3cd4f93fbe5b79dd17086cf57cdda3de34ab501519f9aaecbf22a2b9e47c4292",
        "job-015-6e3b0261dd36d48a3cce927db5f9770800dc38cc7d3863073cd4626bd45fa5c0.json",
        "b26a47478d5ea8bfb18a2de192cae82df0266896d7a7b9a7107ffa970cb1fbd7",
        "0f7bc6abc8f038701799f47bfc6b3a7a43b58bde3b6faaa348e692b625423b2f",
    ),
    (
        "development-calibration-job-5ece1ae1212afbc71958ea1a72678b7beaa0fc62805b7e414260d6a8f12ceda1",
        "job-016-f12e4d6dc354ce50e66ccace65602cc110b909d201370dc338889738b3dac0df.json",
        "a0885bd18ae0d45292b3ed4fd817459bc555e3475429742ce80c3a4dfd963252",
        "79578a26d7c7e9f16c6da2f2132d5a80d96c3e430f90ae86ac152fa95c0759e5",
    ),
    (
        "development-calibration-job-589217e121c0edd60817b0988a571e9e60ebc110b421bb38c98bd44b0c2f9b2f",
        "job-017-150ef842eca4c1131a4b78047751c2ed65bfde79151aeb1987c48ff3d94df28f.json",
        "76bc915351027060b81d81dee5d33153c090eed1baf941e19d3cc1c26aabed97",
        "8e4c6a5169e805a1716da9eb5cf3c681e428b07f89a7f1581727664a1342912b",
    ),
    (
        "development-calibration-job-f56dcca2298611cde0cd26c4b1a1307985c2e9079d1ee387816692c3734f091b",
        "job-018-eab6da69681d166114feffd4121a92280cf4e7e4381d077b82169edbff304f9a.json",
        "d268585f018bdf20d43a5a7a9d0a569882d2c9962f1e2c06a5a556a36067bbfd",
        "17d69da2192a1778946f8145c5b18538630825ff2c49c5e2ba9e5ebdefcc808d",
    ),
    (
        "development-calibration-job-90cd5c19a02a267b5e16d641fe9ca40997278f00217c9fbfa43d1a16942365de",
        "job-019-8be73ec39082c8cebe2ee062adedaf1869ed37dda06ad98d759935709de03132.json",
        "8f28655336fbfb083a0c50c2227dd0e74b18b28d5273a1f8ba2fd6726e9047ed",
        "657784cdea40bbfc32269c4fbe018136199d2123429e4c95af5b586686bde1d0",
    ),
    (
        "development-calibration-job-6c17c51cdb705d0a62172578f7b5183ab50b2e4a36e1d3e84b2f892013cabef0",
        "job-020-2f41e875e9441bd65f747675b63c01e2fe3bf2d20e936893665074d3842ead4f.json",
        "41001c110525832ac9c2c4234d0f9743e9a6ad8220aa6fb82cfff0c16ec5bdef",
        "e7adfb7f266378fca1de13190cad1eb4d12ea03674124077b88e667dc7fabdd6",
    ),
    (
        "development-calibration-job-fc727ad16d639fa44ef9053b13fe4ef6eaaf05c04270b3bc26a3c6ec270b4bff",
        "job-021-349e44dd3691195ba9ce6adb3695050f543da1c4bb8076d16d0043879020b25c.json",
        "1bd79264dcd08ed72211f0c00613c0d066879b8faf28be4d7f48f9a95549d0a3",
        "8c1b67d017bec32556d605cb831a0acf2abc1e5defd0a55d88c0a00e0b390098",
    ),
    (
        "development-calibration-job-c5ce4c4c4f68b8fce897fef3c5085a97b7379f870bf43569d053091ec092af84",
        "job-022-10527bfc81a2ded3e80811d3a98b8f583e3429a14a21cfad06ed96cc7a17a0b3.json",
        "4424670ce732218f6100838fd6b49fe03391236284f58e9c130478aab64c5057",
        "33ec1a1c1fcdf157527a084eb3fcc6c8169744de1f54aef7e8b8e7355c04af0d",
    ),
    (
        "development-calibration-job-9777bb321698f14a761832da3c7be68fa9c53cbd778231445d0888bc5aac70d7",
        "job-023-45227784a79df2226375d1e912761fd36a6f9b7ded3a36be698cd864f6c2c9b4.json",
        "dffc30590287864db2279aa74cab1f2b0ed5a540034958a649ee5385dd4c2a44",
        "a333faab60aceeb9398de8176ff01c80e0a2b335507c164932ab80f5a8b3aae4",
    ),
)


__all__ = [
    "FROZEN_BASE_COMMIT",
    "FROZEN_MANIFEST_FILE_SHA256",
    "FROZEN_MANIFEST_SHA256",
    "FROZEN_SOURCE_CHECKPOINTS",
    "FROZEN_SOURCE_IMPLEMENTATION_SHA256",
    "FROZEN_SOURCE_RUN_ID",
    "FROZEN_SOURCE_RUN_IDENTITY_FILE_SHA256",
    "FROZEN_SOURCE_RUN_IDENTITY_SHA256",
    "FROZEN_SOURCE_SUMMARY_FILE_SHA256",
    "FROZEN_SOURCE_SUMMARY_SHA256",
    "H23_DIAGNOSTIC_CHECKPOINT_SCHEMA",
    "H23_DIAGNOSTIC_MAX_JOBS",
    "H23_DIAGNOSTIC_PLAN_ID",
    "H23_DIAGNOSTIC_PLAN_SCHEMA",
    "H23_DIAGNOSTIC_RUN_SCHEMA",
    "H23_DIAGNOSTIC_SUMMARY_SCHEMA",
    "IMPLEMENTATION_FILES",
    "LEGACY_IMPLEMENTATION_FILES",
    "REGISTERED_GEOMETRY_POLICY_PAYLOAD",
    "REGISTERED_NATIVE_SETTINGS_PAYLOAD",
    "REGISTERED_POLICY_PAYLOAD",
]
