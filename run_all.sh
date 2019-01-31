echo "Model training and saving ..."
python train.py

echo ""
echo "Model restoring ..."
python restore.py

echo ""
echo "Model exporting ..."
python export.py

echo ""
echo "Checking exported model ..."
python load_exported.py

echo ""
echo "Completed"