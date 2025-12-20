// Program agent is a dead simple AI agent.
//
// See https://fly.io/blog/everyone-write-an-agent/ for initial idea.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"log/slog"
	"os"
	"os/exec"
	"time"

	"github.com/chzyer/readline"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
)

func main() {
	// Set up logging.
	f, err := os.OpenFile("agent.log", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	handler := slog.NewTextHandler(f, nil)
	logger := slog.New(handler)
	slog.SetDefault(logger)

	ctx := context.Background()
	client := openai.NewClient()

	// Set up the context. We will add to this each time we go around the main loop.
	msgContext := &msgContext{
		client: client,
		messages: []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage("Do not run commands on the internet as a whole."),
		},
	}
	if err != nil {
		log.Fatal(err)
	}

	// Set up readline.
	rl, err := readline.NewEx(&readline.Config{
		Prompt:      "> ",
		HistoryFile: "history",
	})
	defer rl.Close()

	// Main loop
	for {
		line, err := rl.Readline()
		if err != nil {
			log.Fatal(err)
		}
		start := time.Now()
		// Actually call the model.
		resp, err := msgContext.call(ctx, line)
		if err != nil {
			log.Fatal(err)
		}
		delay := time.Since(start)
		fmt.Println(resp)
		fmt.Printf("waiting for: tools: %s (%2.f%%), LLM: %s (%2.f%%), total: %s. Total calls: LLM %d, tools: %d tokens: %d\n",
			msgContext.toolTime,
			msgContext.toolTime.Seconds()/delay.Seconds()*100,
			msgContext.llmTime,
			msgContext.llmTime.Seconds()/delay.Seconds()*100,
			delay, msgContext.toolCalls, msgContext.llmCalls, msgContext.tokens)
		msgContext.resetStats()

	}
}

// msgContext is the current context sent to the model on each request.
type msgContext struct {
	client   openai.Client
	messages []openai.ChatCompletionMessageParamUnion

	toolCalls int
	toolTime  time.Duration
	llmCalls  int
	llmTime   time.Duration
	tokens    int64
}

func (c *msgContext) resetStats() {
	c.toolCalls = 0
	c.toolTime = 0
	c.llmCalls = 0
	c.llmTime = 0
	c.tokens = 0
}

// call calls the model with the new request (aka line), the current context, and the possible tools.
//
// The model may ask for a tool call. If so, the call calls the tool, then calls the model again with the results of the tool call.
func (c *msgContext) call(ctx context.Context, line string) (string, error) {

	// The set of tools the model can call.
	tools := []openai.ChatCompletionToolParam{
		openai.ChatCompletionToolParam{
			Function: openai.FunctionDefinitionParam{
				Name:        "postgres",
				Description: param.NewOpt("query the postgres db"),
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"query": map[string]interface{}{
							"type":        "string",
							"description": "postgres query to run ",
						},
					},
					"required": []string{"query"},
				},
			},
		},
		openai.ChatCompletionToolParam{
			Function: openai.FunctionDefinitionParam{
				Name:        "shell",
				Description: param.NewOpt("run a shell command"),
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"command": map[string]interface{}{
							"type":        "string",
							"description": "shell command to run",
						},
					},
					"required": []string{"command"},
				},
			},
		},
	}

	// Add the new message from the user to the context.
	c.messages = append(c.messages, openai.UserMessage(line))
	var chatCompletion *openai.ChatCompletion

	cont := true
	for cont {
		cont = false
		msg := openai.ChatCompletionNewParams{
			Messages: c.messages,
			Model:    "gpt-5",
			Tools:    tools,
		}
		var err error

		// Call the model.

		start := time.Now()
		chatCompletion, err = c.client.Chat.Completions.New(ctx, msg)
		if err != nil {
			return "", err
		}
		elapsed := time.Since(start)
		c.llmCalls++
		c.llmTime += elapsed

		// Log token usage and accumulate totals if available
		u := chatCompletion.Usage
		c.tokens += u.TotalTokens
		slog.Info("llm usage",
			"id", chatCompletion.ID,
			"model", chatCompletion.Model,
			"prompt_tokens", u.PromptTokens,
			"completion_tokens", u.CompletionTokens,
			"total_tokens", u.TotalTokens,
			"elapsed", elapsed.String(),
		)

		message := chatCompletion.Choices[0].Message
		mp := message.ToAssistantMessageParam()

		// Append the result to the context.
		c.messages = append(c.messages, openai.ChatCompletionMessageParamUnion{OfAssistant: &mp})

		// If we have a tool call request from the module, first run the tool call, then add it to the model, then call the model again via `cont`.
		for _, toolCall := range message.ToolCalls {
			argsPreview := toolCall.Function.Arguments
			if len(argsPreview) > 300 {
				argsPreview = argsPreview[:300] + "… (truncated)"
			}
			slog.Info("tool call", "id", toolCall.ID, "name", toolCall.Function.Name, "args", argsPreview)
			start = time.Now()
			out, err := c.callTool(toolCall.Function)
			elapsed = time.Since(start)
			// Track tool metrics
			c.toolCalls++
			c.toolTime += elapsed
			slog.Info("tool metrics",
				"calls", c.toolCalls,
				"last_duration", elapsed.String(),
				"total_tool_time", c.toolTime.String())

			// If we got an error... just tell the model.
			if err != nil {
				slog.Error("error in tool call", "err", err)
				out += "error " + err.Error()
			}
			c.messages = append(c.messages, openai.ToolMessage(out, toolCall.ID))
			cont = true
		}
	}

	// Not a tool call. we have the final result.
	return chatCompletion.Choices[0].Message.Content, nil
}

// callTool dispatches the request to one of the tools provided.
func (c *msgContext) callTool(in openai.ChatCompletionMessageToolCallFunction) (string, error) {
	slog.Info("calltool.start", "name", in.Name, "args", in.Arguments)
	args := map[string]string{}
	if err := json.Unmarshal([]byte(in.Arguments), &args); err != nil {
		slog.Error("failed to unmarsh", "err", err)
		return "", err
	}
	switch in.Name {
	case "postgres":
		return c.postgres(args)
	case "shell":
		return c.shell(args)
	default:
		return "", errors.New("no tool")
	}
}

func (c *msgContext) postgres(args map[string]string) (string, error) {
	query := args["query"]
	cmd := exec.Command("psql", "postgres", "-c", query)
	fmt.Println(query)
	var stdout bytes.Buffer
	cmd.Stdout = &stdout
	err := cmd.Run()
	out := stdout.String()
	slog.Info("command finished", "exit_err", err, "bytes", len(out))
	return out, err
}

func (c *msgContext) shell(args map[string]string) (string, error) {
	slog.Info("running command", "args", args)
	query := args["command"]
	cmd := exec.Command("/bin/bash", "-c", query)
	fmt.Println(query)
	var stdout bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stdout
	err := cmd.Run()
	out := stdout.String()
	slog.Info("command finished", "exit_err", err, "bytes", len(out))
	return out, err
}
