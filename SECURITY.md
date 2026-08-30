# Security Policy

## Supported versions

Formwork is early-stage. Fixes are released from the latest published version
on PyPI; older versions are not patched.

| Version | Supported |
| ------- | --------- |
| latest  | yes       |
| older   | no        |

## Reporting a vulnerability

Please report suspected vulnerabilities privately, through GitHub's
[private vulnerability reporting](https://github.com/ekberglpnar/formwork/security/advisories/new)
form. Do not open a public issue for a security problem.

A useful report describes what an attacker can do, the version you tested, and
the smallest set of steps that reproduces it. You should get an initial reply
within a week.

## Scope notes

Formwork sits between an application and a language model, so two areas are
worth calling out explicitly:

**Model output is untrusted input.** Formwork validates it structurally and
then semantically, but it does not sanitise it for downstream use. Escaping
values before they reach a shell, a query or a template is the caller's job.

**Rule and violation text reaches the model.** Violation messages are written
into the repair prompt by design. Putting secrets, credentials or personal data
into a rule message sends them to the provider on the next repair attempt.
