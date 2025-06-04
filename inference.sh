curl -X POST http://localhost:8000/start-inference-task\
    -H 'Content-Type: application/json'\
    -d '{"raw_input": "data_Li1Mn1O2",
         "generate_cif": "true"
         }'\
