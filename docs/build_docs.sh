#!/bin/bash
# Utility script for building arte documentation

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Arte Documentation Builder${NC}"
echo ""

# Check if we're in the docs directory
if [ ! -f "conf.py" ]; then
    echo -e "${RED}Error: This script must be run from the docs/ directory${NC}"
    exit 1
fi

# Function to check dependencies
check_deps() {
    echo -e "${YELLOW}Checking documentation dependencies...${NC}"
    python -c "import sphinx" 2>/dev/null || {
        echo -e "${RED}Sphinx is not installed. Installing dependencies...${NC}"
        pip install -r requirements.txt
    }
    echo -e "${GREEN}✓ Dependencies OK${NC}"
    echo ""
}

# Function to clean build
clean_build() {
    echo -e "${YELLOW}Cleaning previous build...${NC}"
    make clean
    echo -e "${GREEN}✓ Clean complete${NC}"
    echo ""
}

# Function to build HTML
build_html() {
    echo -e "${YELLOW}Building HTML documentation...${NC}"
    make html
    echo ""
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Build successful!${NC}"
        echo -e "Documentation is in: ${GREEN}_build/html/index.html${NC}"
        echo ""
        
        # Ask if user wants to open
        read -p "Open documentation in browser? (y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            if command -v open &> /dev/null; then
                open _build/html/index.html
            elif command -v xdg-open &> /dev/null; then
                xdg-open _build/html/index.html
            else
                echo "Please open _build/html/index.html in your browser"
            fi
        fi
    else
        echo -e "${RED}✗ Build failed${NC}"
        exit 1
    fi
}

# Parse command line arguments
case "${1:-build}" in
    check)
        check_deps
        ;;
    clean)
        clean_build
        ;;
    build)
        check_deps
        clean_build
        build_html
        ;;
    quick)
        echo -e "${YELLOW}Quick build (no clean)...${NC}"
        make html
        ;;
    help)
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  build  - Clean and build documentation (default)"
        echo "  quick  - Build without cleaning"
        echo "  clean  - Clean build artifacts"
        echo "  check  - Check dependencies"
        echo "  help   - Show this help"
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo "Run '$0 help' for usage information"
        exit 1
        ;;
esac
