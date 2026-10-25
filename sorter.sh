```bash
#!/bin/bash
cd "$(dirname "$0")"

sudo pigpiod
source ~/waste-env/bin/activate

GPIOZERO_PIN_FACTORY=pigpio python3 sorter.py
```
