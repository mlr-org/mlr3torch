"""Translate a (simple) Dockerfile into a shell script that replays its steps inside a container.

Supports FROM (printed as a comment), ENV, ARG, COPY (from /build/context), RUN and CMD (ignored).
Each RUN is executed with `sh -c`, like Docker; the build stops at the first failing step.
ENV values are exported and appended to /etc/environment, which enroot reads when the container starts.
"""
import shlex
import sys

lines = open(sys.argv[1]).read().split("\n")
# join continuation lines and drop comments
instructions, cur = [], ""
for line in lines:
    # comment lines are skipped, also inside of continued instructions (as Docker does)
    if line.strip().startswith("#") or (not cur and not line.strip()):
        continue
    if line.rstrip().endswith("\\"):
        cur += line.rstrip()[:-1]  # Docker joins continued lines by removing the backslash and the line break
        continue
    cur += line
    instructions.append(cur)
    cur = ""

out = ["#!/bin/bash", "set -e", "cd /", ""]
for ins in instructions:
    cmd, _, rest = ins.strip().partition(" ")
    cmd = cmd.upper()
    if cmd == "FROM":
        out.append(f"# FROM {rest.strip()}")
    elif cmd in ("ENV", "ARG"):
        for kv in shlex.split(rest.replace("\n", " ")):
            k, _, v = kv.partition("=")
            out.append(f'export {k}="{v}"')
            if cmd == "ENV":
                out.append(f'echo "{k}=${{{k}}}" >> /etc/environment')
    elif cmd == "COPY":
        src, dst = rest.split()
        out.append(f"cp -r /build/context/{src} {dst}")
    elif cmd == "RUN":
        out.append("echo " + shlex.quote("##### STEP: RUN " + rest.strip().splitlines()[0][:100]))
        out.append("sh -c " + shlex.quote(rest.strip()))
    elif cmd == "CMD":
        continue
    else:
        sys.exit(f"unsupported instruction: {cmd}")
out.append("echo '##### BUILD DONE'")
print("\n".join(out))
