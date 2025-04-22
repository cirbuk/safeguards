# Agent Safety Dashboard

A modern web dashboard for monitoring and managing agent safety controls, built with Next.js and Chakra UI.

## Features

- Real-time metrics visualization
- Budget pool management
- Agent monitoring and control
- Resource usage analytics
- Alert configuration
- Dark/light mode support

## Getting Started

### Prerequisites

- Node.js 18.x or later
- npm or yarn
- Running instance of Agent Safety Framework API

### Installation

1. Install dependencies:
```bash
npm install
# or
yarn install
```

2. Set up environment variables:
```bash
cp .env.example .env.local
```

Edit `.env.local` with your configuration:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

3. Run the development server:
```bash
npm run dev
# or
yarn dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Project Structure

```
frontend/
├── src/
│   ├── api/          # API client and types
│   ├── components/   # Reusable components
│   ├── pages/        # Next.js pages
│   ├── styles/       # Global styles
│   └── utils/        # Utility functions
├── public/           # Static assets
└── package.json      # Dependencies and scripts
```

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm start` - Start production server
- `npm run lint` - Run ESLint
- `npm run type-check` - Run TypeScript type checking

## Development

### Adding New Features

1. Create new components in `src/components/`
2. Add new pages in `src/pages/`
3. Add API endpoints in `src/api/`
4. Update types as needed

### Code Style

- Use TypeScript for type safety
- Follow ESLint configuration
- Use Chakra UI components for consistency
- Follow component-based architecture

## Deployment

1. Build the application:
```bash
npm run build
```

2. Start the production server:
```bash
npm start
```

## Integration with Agent Safety Framework

The dashboard integrates with the Agent Safety Framework API to provide:

- Real-time metrics visualization
- Budget pool management
- Agent monitoring
- Resource usage analytics
- Alert configuration

### API Integration

The dashboard communicates with the Agent Safety Framework API through:

- WebSocket connections for real-time updates
- REST API endpoints for CRUD operations
- Authentication via JWT tokens

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
