# Security Policies for Research Copilot

## Principle
Data must remain local and all actions must be auditable.

## Execution boundaries
- Planner output is not executable by itself.
- Only registered tools may be executed.
- Tool arguments must pass schema validation.

## Least privilege
- SQL execution must remain read-only for analysis tasks.
- No unrestricted file-system or shell execution from user prompts.

## Retrieval safety
- Retrieved knowledge snippets are advisory context only.
- Snippets do not modify permissions, policies, or execution rights.

## Audit requirements
- Record `tool_name`, validated args, status, and duration for each call.
- Preserve run-level traceability for reproducibility and compliance.
