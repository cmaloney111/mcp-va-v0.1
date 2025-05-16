# MCP Vision Agent (v0.1)

This package provides the Vision Agent for use with the MCP framework.

## Installation & Build

1. Clone the repository:

   ```bash
   git clone https://github.com/cmaloney111/mcp-va-v0.1.git
   ```

2. Navigate into the project directory:

   ```bash
   cd mcp-va-v0.1
   ```

3. Build the project:

   ```bash
   npm run build
   ```

## Client Configuration

After building, configure your MCP client with the following settings:

```json
{
  "mcpServers": {
    "Vision Agent": {
      "command": "node",
      "args": [
        "/path/to/build/index.js"
      ],
      "env": {
        "VISION_AGENT_API_KEY": "YOUR_API_KEY_HERE",
        "INPUT_DIRECTORY": "../input",
        "OUTPUT_DIRECTORY": "../output",
        "IMAGE_DISPLAY_ENABLED": "true"
      }
    }
  }
}
```

> **Note:** Replace `/path/to/build/index.js` with the actual path to your built `index.js` file, and set your environment variables as needed.
