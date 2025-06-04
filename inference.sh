curl -X POST http://localhost:8000/start-inference-task\
    -H 'Content-Type: application/json'\
    -d '{"model_path": "models/crystallm_v1_small/ckpt.pt",
         "model_url": "https://zenodo.org/records/10642388/files/crystallm_v1_small.tar.gz",
         "raw_input": "data_Li1Mn1O2",
         "attempts": "1"
         }'\
