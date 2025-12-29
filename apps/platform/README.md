# EMG Task Playground

A browser-based task playground to elicit repeatable mouse/trackpad interactions (click, drag, scroll) for EMG dataset collection.

## Setup

Install dependencies:

```bash
bun install
```

## Development

Start development server with HMR:

```bash
bun dev
```

Then open `http://localhost:3000`.

## Production

Run in production mode:

```bash
bun start
```

Build for deployment:

```bash
bun run build
```

## Code Quality

This project uses [Biome](https://biomejs.dev/) for formatting and linting.

Format code:

```bash
bun run format        # Format and write
bun run format:check  # Check only
```

Lint code:

```bash
bun run lint          # Lint and fix
bun run lint:check    # Check only
```

Run both format + lint:

```bash
bun run check         # Format + lint with fixes
bun run check:ci      # Check only (for CI)
```

VSCode will automatically format on save if you have the [Biome extension](https://marketplace.visualstudio.com/items?itemName=biomejs.biome) installed.

