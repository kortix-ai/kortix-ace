"""CLI entry point for ace-learn-opencode.

Commands:
    ace-learn-opencode                    # Learn from latest session
    ace-learn-opencode --session <id>     # Learn from specific session
    ace-learn-opencode --url <url>        # Custom OpenCode API URL
    ace-learn-opencode --user <id>        # User ID for scoped skillbooks
    ace-learn-opencode --model <model>    # LLM model override

    ace-learn-opencode insights           # Show learned strategies
    ace-learn-opencode remove <id>        # Remove a specific insight
    ace-learn-opencode clear --confirm    # Clear all insights
    ace-learn-opencode doctor             # Check prerequisites
    ace-learn-opencode inject             # Regenerate SKILL.md from skillbook
"""

import logging
import sys

from .client import DEFAULT_OPENCODE_URL


def cmd_learn(args):
    """Learn from an OpenCode session."""
    from .learner import OpenCodeLearner

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    learner = OpenCodeLearner(
        opencode_url=args.url,
        workspace_path=args.workspace,
        user_id=args.user,
        project_id=args.project,
        model=args.model,
    )

    if args.session:
        print(f"Learning from session: {args.session}")
        success = learner.learn_from_session(args.session, force=args.force)
    else:
        print("Learning from latest session...")
        success = learner.learn_from_latest(project_id=args.project)

    if success:
        print("\nLearning complete!")
        insights = learner.get_insights(args.project)
        print(f"Skillbook now has {insights['count']} strategies.")
    else:
        print("\nLearning failed.", file=sys.stderr)
        sys.exit(1)


def cmd_insights(args):
    """Show current ACE learned strategies."""
    from .learner import OpenCodeLearner

    learner = OpenCodeLearner(
        opencode_url=args.url,
        workspace_path=args.workspace,
        user_id=args.user,
    )

    insights = learner.get_insights(args.project)

    if not insights["skills"]:
        print("No insights yet. Run /ace-learning after a session to start learning.")
        return

    print(f"ACE Learned Strategies ({insights['count']} total)\n")

    for section, section_skills in sorted(insights["sections"].items()):
        print(f"## {section.replace('_', ' ').title()}")
        for s in section_skills:
            score = f"({s.helpful}↑ {s.harmful}↓)"
            print(f"  [{s.id}] {s.content} {score}")
        print()


def cmd_remove(args):
    """Remove a specific insight by ID."""
    from .learner import OpenCodeLearner

    learner = OpenCodeLearner(
        opencode_url=args.url,
        workspace_path=args.workspace,
        user_id=args.user,
    )

    if learner.remove_insight(args.id, project_id=args.project):
        print(f"Removed insight matching: {args.id}")
    else:
        print(f"No insight found matching '{args.id}'")
        print("Use 'ace-learn-opencode insights' to see available insights.")


def cmd_clear(args):
    """Clear all ACE learned strategies."""
    if not args.confirm:
        print("This will delete all learned strategies.")
        print("Run with --confirm to proceed: ace-learn-opencode clear --confirm")
        return

    from .learner import OpenCodeLearner

    learner = OpenCodeLearner(
        opencode_url=args.url,
        workspace_path=args.workspace,
        user_id=args.user,
    )
    learner.clear_insights(project_id=args.project)
    print("All insights cleared. ACE will start fresh.")


def cmd_inject(args):
    """Regenerate SKILL.md from the current skillbook."""
    from .learner import OpenCodeLearner

    learner = OpenCodeLearner(
        opencode_url=args.url,
        workspace_path=args.workspace,
        user_id=args.user,
    )

    path = learner.inject_skill(project_id=args.project)
    if path:
        print(f"SKILL.md written to: {path}")
    else:
        print("No strategies to write. Run /ace-learning after a session first.")


def cmd_doctor(args):
    """Verify ACE prerequisites and configuration."""
    from .client import OpenCodeClient

    print("ACE Doctor (OpenCode) — Checking prerequisites\n")
    all_ok = True

    # 1. Check OpenCode API
    print(f"1. OpenCode API ({args.url})...")
    client = OpenCodeClient(base_url=args.url)
    if client.health_check():
        print("   OK: API is reachable")
        sessions = client.list_sessions()
        print(f"   OK: {len(sessions)} sessions found")
        if sessions:
            latest = sessions[0]
            print(f"   Latest: {latest.get('id', 'unknown')[:8]}... ({latest.get('title', 'untitled')})")
    else:
        print("   FAIL: API not reachable")
        print(f"   Check that OpenCode is running at {args.url}")
        all_ok = False

    # 2. Check LLM availability
    print("\n2. LLM availability...")
    try:
        import litellm
        print(f"   OK: litellm installed (v{litellm.__version__})")

        # Check for API keys
        has_key = any(
            os.environ.get(k)
            for k in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"]
        )
        if has_key:
            print("   OK: API key found in environment")
        else:
            print("   WARN: No API key found (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)")
            print("   Set an API key for the model you want to use")
    except ImportError:
        print("   FAIL: litellm not installed")
        all_ok = False

    # 3. Check TOON availability
    print("\n3. TOON compression...")
    try:
        from toon import encode
        print("   OK: python-toon installed")
    except ImportError:
        print("   WARN: python-toon not installed (will use JSON fallback)")
        print("   Install with: pip install python-toon")

    # 4. Check output paths
    print(f"\n4. Output paths (workspace: {args.workspace})...")
    from pathlib import Path

    workspace = Path(args.workspace)
    ace_dir = workspace / ".kortix" / "ace"
    skill_dir = workspace / "opt" / "opencode" / "skills" / "KORTIX-ace"
    memory_dir = workspace / ".kortix" / "memory"

    for name, path in [("ACE data", ace_dir), ("SKILL.md", skill_dir), ("Memory", memory_dir)]:
        if path.exists():
            print(f"   OK: {name} dir exists ({path})")
        else:
            print(f"   INFO: {name} dir will be created ({path})")

    # Summary
    print("\n" + "=" * 50)
    if all_ok:
        print("All checks passed! ACE is ready to learn.")
        print("\nTo learn from your latest session: ace-learn-opencode")
    else:
        print("Some checks failed. Please fix the issues above.")

    return 0 if all_ok else 1


def main():
    """CLI entry point for ace-learn-opencode."""
    import argparse

    parser = argparse.ArgumentParser(
        description="ACE learning for OpenCode — learns from autonomous agent sessions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  ace-learn-opencode                      Learn from latest session
  ace-learn-opencode --session <id>       Learn from specific session
  ace-learn-opencode insights             Show learned strategies
  ace-learn-opencode remove <id>          Remove a specific insight
  ace-learn-opencode clear --confirm      Clear all insights
  ace-learn-opencode doctor               Check prerequisites
  ace-learn-opencode inject               Regenerate SKILL.md from skillbook
""",
    )

    # Global flags
    parser.add_argument(
        "--url", default=DEFAULT_OPENCODE_URL,
        help=f"OpenCode API URL (default: {DEFAULT_OPENCODE_URL})",
    )
    parser.add_argument(
        "--workspace", "-w", default="/workspace",
        help="Workspace root path (default: /workspace)",
    )
    parser.add_argument(
        "--user", "-u", default=None,
        help="User ID for scoped skillbooks (default: from ACE_USER_ID env or 'default')",
    )
    parser.add_argument(
        "--project", "-p", default=None,
        help="Project ID (default: auto-detected from session)",
    )
    parser.add_argument(
        "--model", "-m", default="anthropic/claude-sonnet-4-20250514",
        help="LLM model for Reflector/SkillManager",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command")

    # Subcommands
    subparsers.add_parser("doctor", help="Check prerequisites")
    subparsers.add_parser("inject", help="Regenerate SKILL.md from skillbook")

    insights_parser = subparsers.add_parser("insights", help="Show learned strategies")
    insights_parser.add_argument("--project", "-p", default=None, dest="project")

    remove_parser = subparsers.add_parser("remove", help="Remove a specific insight")
    remove_parser.add_argument("id", help="Insight ID or keyword to match")

    clear_parser = subparsers.add_parser("clear", help="Clear all insights")
    clear_parser.add_argument("--confirm", action="store_true", help="Confirm clearing")

    # Main learning flags
    parser.add_argument("--session", "-s", default=None, help="Session ID to learn from")
    parser.add_argument("--force", "-f", action="store_true", help="Force relearning")

    args = parser.parse_args()

    if args.command == "doctor":
        sys.exit(cmd_doctor(args))
    elif args.command == "insights":
        cmd_insights(args)
    elif args.command == "remove":
        cmd_remove(args)
    elif args.command == "clear":
        cmd_clear(args)
    elif args.command == "inject":
        cmd_inject(args)
    else:
        cmd_learn(args)


if __name__ == "__main__":
    main()
