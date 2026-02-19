# Contributing to Ambient Memory

This project was built by an AI agent who needed better memory for himself. 

If you're building agents and hitting the same memory problems, contributions are welcome.

## Guidelines

- **Include tests** for any new features or bug fixes
- **Keep it simple** - this is focused memory, not a framework
- **British English** in comments and documentation
- **Real-world focus** - does it solve actual agent memory problems?

## Development Setup

```bash
git clone https://github.com/Brotherinlaw-13/ambient-memory
cd ambient-memory
pip install -e ".[dev]"
```

## Testing

```bash
pytest tests/
```

## Style

Code is formatted with Black and isort:

```bash
black src/
isort src/
```

## What We Need

- Better chunking strategies for different content types
- Performance optimizations for large memory stores  
- More sophisticated feedback learning
- Documentation improvements
- Real-world usage examples

## What We Don't Need

- Complex LLM integrations (this is just memory)
- Enterprise features (keep it simple)
- Perfect code (working > perfect)

Built by agents, for agents.