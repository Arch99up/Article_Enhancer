#!/bin/bash
gunicorn -w 1 -b 0.0.0.0:$PORT --timeout 60 app:app
