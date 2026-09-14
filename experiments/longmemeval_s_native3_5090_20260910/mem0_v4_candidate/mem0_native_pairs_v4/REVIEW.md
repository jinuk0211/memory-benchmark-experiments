Mem0 v2: approved by independent Python reviewer.

Eight observation tests passed; seven actual unmodified native method AST cases passed independently. Same vendor, SDK parameters, prompts, pair ingestion and QA. Only authenticated, response-supported native caught parsing/shape warnings are permitted and counted; transport/storage failures remain fatal. Original SDK responses and finish reasons saved in compressed JSONL. Required native_degradation.json and response capture are sealed and reverified.

This is a new source/protocol version; v1 completed6 and failure1 remain separate. No scoring/generalization claim from CPU tests.
