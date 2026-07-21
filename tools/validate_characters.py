#!/usr/bin/env python3
"""
Validate setup/characters/*.txt against the rules the Imperator engine actually
enforces at game setup. Catches the errors that otherwise only show up as
jomini_script_system.cpp / charactereffectimpl.cpp spam in the game log.

Rules (all derived from vanilla, see game/setup/characters/):

  R1  A character whose death_date is on or before the start date is DEAD when
      setup effects run. Dead characters must not carry add_gold,
      add_popularity, marry_character or give_office. Across all 76
      dead-before-start characters in vanilla, zero carry any of these.
      Violations produce: "add_gold : scope was dead", "add_popularity : scope
      was dead", "marry_character : scope was dead".

  R2  marry_character=char:N requires N to exist, to be ALIVE at start, to have
      a LOWER id than the referencing character (the engine instantiates in
      ascending global id order), and to be the opposite gender.
      Violations produce: "Tried to arrange a marriage with an invalid
      character: N for M" or "marry_character : target was dead".

  R3  set_as_ruler=char:N must name the character whose own block contains it,
      and that character must be alive at start. Vanilla does this exactly once
      per country file, always as a self-reference from inside the ruler's own
      block. Violations produce: "set_as_ruler effect [ Target Character ... is
      not alive ]" - or, worse, silently install another country's ruler.

  R4  father=/mother=char:N must exist and have a LOWER id (same ascending-id
      reason as R2).

Usage:
    python tools/validate_characters.py                # validate the mod
    python tools/validate_characters.py --dir <path>   # validate another tree
    python tools/validate_characters.py --start 1001.1.1

Exit code is 0 when clean, 1 when any violation is found, so it can be wired
into a pre-commit hook or run alongside imperator-tiger.exe.
"""

import argparse
import os
import re
import sys

DEFAULT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "setup", "characters",
)
DEFAULT_START = (1001, 1, 1)

# Effects that only make sense on a character who is alive when setup runs.
LIVING_ONLY = ("add_gold", "add_popularity", "marry_character", "give_office")


def parse_date(text):
    m = re.match(r"(\d+)\.(\d+)\.(\d+)", text.strip().strip('"'))
    return tuple(int(x) for x in m.groups()) if m else None


def load_characters(directory):
    """Return {char_id: block}. Blocks are the `<id>={ ... }` entries that sit
    one level deep inside each file's top-level "TAG"={ ... } wrapper."""
    chars = {}
    duplicates = []
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".txt") or filename.startswith("000_"):
            continue
        path = os.path.join(directory, filename)
        with open(path, encoding="utf-8-sig") as handle:
            lines = handle.read().split("\n")

        depth = 0
        i = 0
        while i < len(lines):
            header = re.match(r"^(\d+)\s*=\s*\{", lines[i].strip())
            if header and depth == 1:
                cid = int(header.group(1))
                start_line = i
                brace = 0
                body_lines = []
                while i < len(lines):
                    brace += lines[i].count("{") - lines[i].count("}")
                    body_lines.append(lines[i])
                    i += 1
                    if brace == 0:
                        break
                body = "\n".join(body_lines)
                death = re.search(r"death_date\s*=\s*(\S+)", body)
                name = re.search(r'first_name\s*=\s*"?([^"\n]+)', body)
                entry = {
                    "id": cid,
                    "file": filename,
                    "line": start_line + 1,
                    "body": body,
                    "death": parse_date(death.group(1)) if death else None,
                    "female": "female=yes" in body.replace(" ", ""),
                    "name": name.group(1).strip() if name else "?",
                }
                if cid in chars:
                    duplicates.append((cid, chars[cid], entry))
                chars[cid] = entry
                continue
            depth += lines[i].count("{") - lines[i].count("}")
            i += 1
    return chars, duplicates


def validate(directory, start):
    chars, duplicates = load_characters(directory)
    problems = []

    def add(entry, rule, message):
        problems.append((entry["file"], entry["line"], entry["id"], rule, message))

    def is_dead(cid):
        entry = chars.get(cid)
        return bool(entry and entry["death"] is not None and entry["death"] <= start)

    for cid, first, second in duplicates:
        problems.append((
            second["file"], second["line"], cid, "R0",
            "duplicate character id, also defined in %s:%d"
            % (first["file"], first["line"]),
        ))

    for cid in sorted(chars):
        entry = chars[cid]
        body = entry["body"]
        dead = is_dead(cid)

        # R1 - dead characters carrying living-only effects
        if dead:
            for key in LIVING_ONLY:
                if re.search(r"^\s*" + key + r"\s*=", body, re.M):
                    add(entry, "R1",
                        "dead at start (death_date=%s) but has %s"
                        % (".".join(str(p) for p in entry["death"]), key))

        # R2 - marriage targets
        for m in re.finditer(r"marry_character\s*=\s*\"?char:(\d+)", body):
            target = int(m.group(1))
            if target not in chars:
                add(entry, "R2", "marry_character=char:%d does not exist" % target)
                continue
            if target >= cid:
                add(entry, "R2",
                    "marry_character=char:%d is not a lower id than %d"
                    % (target, cid))
            if chars[target]["female"] == entry["female"]:
                add(entry, "R2",
                    "marry_character=char:%d (%s) is the same gender"
                    % (target, chars[target]["name"]))
            if is_dead(target) and not dead:
                add(entry, "R2",
                    "marry_character=char:%d (%s) is dead at start"
                    % (target, chars[target]["name"]))

        # R3 - ruler assignment
        for m in re.finditer(r"set_as_ruler\s*=\s*\"?char:(\d+)", body):
            target = int(m.group(1))
            if target != cid:
                who = chars[target]["name"] if target in chars else "MISSING"
                where = chars[target]["file"] if target in chars else "?"
                add(entry, "R3",
                    "set_as_ruler=char:%d (%s, %s) inside block %d - "
                    "should be a self-reference to char:%d"
                    % (target, who, where, cid, cid))
            elif dead:
                add(entry, "R3",
                    "set_as_ruler=char:%d but that character is dead at start "
                    "(death_date=%s)"
                    % (cid, ".".join(str(p) for p in entry["death"])))

        # R4 - parents
        for key in ("father", "mother"):
            for m in re.finditer(key + r"\s*=\s*\"?char:(\d+)", body):
                target = int(m.group(1))
                if target not in chars:
                    add(entry, "R4", "%s=char:%d does not exist" % (key, target))
                elif target >= cid:
                    add(entry, "R4",
                        "%s=char:%d is not a lower id than %d" % (key, target, cid))

    return chars, problems


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default=DEFAULT_DIR,
                        help="setup/characters directory to validate")
    parser.add_argument("--start", default="1001.1.1",
                        help="game start date, default 1001.1.1")
    args = parser.parse_args()

    start = parse_date(args.start)
    if start is None:
        print("bad --start date: %s" % args.start, file=sys.stderr)
        return 2

    chars, problems = validate(args.dir, start)
    print("checked %d characters in %s (start %s)"
          % (len(chars), args.dir, args.start))

    if not problems:
        print("OK - no violations")
        return 0

    by_rule = {}
    for problem in problems:
        by_rule.setdefault(problem[3], []).append(problem)

    for rule in sorted(by_rule):
        rows = by_rule[rule]
        print("\n%s - %d violation(s)" % (rule, len(rows)))
        for filename, line, cid, _, message in rows:
            print("  %s:%d  char %d: %s" % (filename, line, cid, message))

    print("\n%d violation(s) total" % len(problems))
    return 1


if __name__ == "__main__":
    sys.exit(main())
