My Project
------------
### index.html

Where you'll write the content of your website. 

### style.css

CSS files add styling rules to your content.

### script.js

If you're feeling fancy you can add interactivity to your site with JavaScript.

### assets

Drag in `assets`, like images or music, to add them to your project

### env

Contains the secret keys that out bot needs for connecting the WebHook to the Telegram infrastructure.

### server.py

The code is connecting the bot specific code with the WebHook calls.

### bot.py

The bot logic triggered by the server.py that interacts with the Unsplash API.

### glitch.json

Will make your app to be set up as a custom app by Glitch, allowing to install scripts.
In our case, we will install Python3 and spin a process that will re-run when changes are made to the *.py source files.
