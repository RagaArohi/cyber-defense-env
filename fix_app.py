with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove the demo route and everything after it
idx = content.find('@app.get("/demo"')
if idx != -1:
    content = content[:idx].rstrip() + '\n'

# Remove any broken main() at the end
import re
content = re.sub(r'\ndef main\(\):\s*$', '', content).rstrip() + '\n'

# Add clean complete main() at the end
content += '''

def main():
    import uvicorn
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level=cfg.log_level)


if __name__ == "__main__":
    main()
'''

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Done!')