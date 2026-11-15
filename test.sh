#!/bin/zsh

case "$1" in
	ollama)
	curl -v "http://localhost:11434/api/chat" \
		-H "Content-Type: application/json" \
		-d '{
		"model": "qwen3.8:27b-dflash2-nvfp4",
		"messages": [
		{
		"role": "user",
		"content": "Why is the sky blue?"
		}
		],
		"stream": false
		}'
	;;
	openrouter)
	curl -v "https://openrouter.ai/api/v1/chat/completions" \
		-H "Content-Type: application/json" \
		-H "Authorization: Bearer $OPENROUTER_API_KEY" \
		-d '{
		"model": "qwen/qwen3.8-27b",
		"messages": [
		{
		"role": "user",
		"content": "Why is the sky blue?"
		}
		],
		"stream": false
		}'
	;;
	*)
	echo "Usage: $0 {ollama|openrouter}" >&2
       exit 1
       ;;
esac


#| Element | Purpose |
#|---|---|
#| `case "$word" in` | Opens the construct. Quote the variable. |
#| `pattern)` | A **glob** pattern followed by `)`. Not a regex. |
#| `\|` | Separates multiple patterns for the same branch. |
#| `;;` | Closes a branch and **prevents fall-through**. |
#| `*)` | Wildcard / default (runs if nothing else matched). |
#| `esac` | Closes the construct (it's `case` backwards). |
