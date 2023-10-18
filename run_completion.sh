curl http://localhost:8000/v1/completions \
	-H "Content-Type: application/json" \
	-d '{
	"model": "NousResearch/Nous-Hermes-llama-2-7b"
	"prompt": "San Francisco is a",
	"max_tokens": 7,
	"temperature": 0
	}'
