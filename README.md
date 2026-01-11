# Agent

This is a dead simple AI agent program. It is based on [You should write an
agent](https://fly.io/blog/everyone-write-an-agent/). I find it useful for
simple stuff.

## Running.

You need a OpenAI API Key.  You then run with

```sh
% export OPENAI_API_KEY=sk-proj-...
% go run agent
```

```sh
OPENAI_API_KEY=$(op item get openai-key --reveal  --fields credential) go run .
```

It prints out each tool as it runs it.

It can answer many questions, for example:

* Which are the largest Postgres tables?
* How long has this computer been up for?
* Are there upgrades to install?
* Can you add a theme to this website?
* Are there insecure devices on this VLAN?

## Getting a OpenAI key

Key idea: you can do a lot with 10 quid's worth of tokens.

1. Go to [OpenAI API](https://openai.com/index/openai-api/), sign in to API
   platform (not chat GPT).
2. Go to
   [billing](https://platform.openai.com/settings/organization/billing/overview),
   select "pay as you go", give them 10 quid.
3. Set `export OPENAI_API_KEY=`


The [Usage console](https://platform.openai.com/settings/organization/usage) may
give an idea of current usage.

## Safety

There is none. My advice would be to run it on a machine you would be happy to
wipe.

## Debugging

* The program provides no output apart from the tools being run and the final
  result.
* There is a chatty JSON log in `agent.log` and also history in `history`.
