#!/bin/bash
# Setup NGC PyTorch as default environment for CENTaUR project

set -e

PROJECT_DIR="/home/juke/git/CENTaUR"
BASHRC_FILE="$HOME/.bashrc"
CENTAUR_BASHRC="$PROJECT_DIR/.bashrc_centaur"

echo "=========================================="
echo "CENTaUR NGC PyTorch Environment Setup"
echo "=========================================="
echo ""

# Step 1: Add to .bashrc
echo "1. Configuring shell environment..."
if grep -q "bashrc_centaur" "$BASHRC_FILE"; then
    echo "   ⚠️  Already configured in .bashrc"
else
    echo "" >> "$BASHRC_FILE"
    echo "# CENTaUR NGC PyTorch Auto-load" >> "$BASHRC_FILE"
    echo "if [ -f $CENTAUR_BASHRC ]; then" >> "$BASHRC_FILE"
    echo "    source $CENTAUR_BASHRC" >> "$BASHRC_FILE"
    echo "fi" >> "$BASHRC_FILE"
    echo "   ✅ Added to .bashrc"
fi

# Step 2: Handle native venv
echo ""
echo "2. Managing native Python virtual environments..."
echo ""
echo "   You have two options:"
echo "   a) Rename venv → venv.backup (safe, can restore)"
echo "   b) Delete venv completely (permanent)"
echo "   c) Keep both (you choose manually)"
echo ""
read -p "   Choose (a/b/c): " choice

case $choice in
    a)
        if [ -d "$PROJECT_DIR/venv" ]; then
            mv "$PROJECT_DIR/venv" "$PROJECT_DIR/venv.backup"
            echo "   ✅ venv renamed to venv.backup"
        else
            echo "   ⚠️  venv not found"
        fi
        ;;
    b)
        if [ -d "$PROJECT_DIR/venv" ]; then
            read -p "   Are you sure? This is permanent (y/n): " confirm
            if [ "$confirm" = "y" ]; then
                rm -rf "$PROJECT_DIR/venv"
                echo "   ✅ venv deleted"
            else
                echo "   ⚠️  Cancelled"
            fi
        else
            echo "   ⚠️  venv not found"
        fi
        ;;
    c)
        echo "   ✅ Keeping both environments"
        ;;
    *)
        echo "   ⚠️  Invalid choice, keeping both"
        ;;
esac

# Step 3: Handle dgx-venv (CPU-only, not useful)
echo ""
if [ -d "$PROJECT_DIR/dgx-venv" ]; then
    echo "3. dgx-venv is CPU-only PyTorch (not useful for GPU training)"
    read -p "   Delete dgx-venv? (y/n): " delete_dgx
    if [ "$delete_dgx" = "y" ]; then
        rm -rf "$PROJECT_DIR/dgx-venv"
        echo "   ✅ dgx-venv deleted"
    else
        echo "   ✅ Keeping dgx-venv"
    fi
fi

# Step 4: Create NGC activation script
echo ""
echo "4. Creating quick-start scripts..."
cat > "$PROJECT_DIR/activate-ngc.sh" << 'EOF'
#!/bin/bash
# Quick activation of NGC environment (without full shell reload)
source /home/juke/git/CENTaUR/.bashrc_centaur
EOF
chmod +x "$PROJECT_DIR/activate-ngc.sh"
echo "   ✅ Created activate-ngc.sh"

echo ""
echo "=========================================="
echo "✅ NGC Environment Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Reload shell: source ~/.bashrc"
echo "  2. Or use: source activate-ngc.sh"
echo "  3. Test: centaur-python --version"
echo "  4. Train: ./scripts/train_qwen25_ngc.sh"
echo ""
echo "Available commands:"
echo "  centaur-python script.py   - Run Python in NGC container"
echo "  centaur-shell              - Interactive NGC shell"
echo "  centaur-run script.py      - Run with logging"
echo "  cdcentaur                  - Go to project directory"
echo ""
