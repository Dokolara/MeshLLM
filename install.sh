APK="/home/foor/MeshLLM/app/build/outputs/apk/debug/app-debug.apk"
  
for dev in $(adb devices | awk 'NR>1 && $2=="device" {print $1}'); do
    echo "Installing on $dev..."
    adb -s "$dev" install -r "$APK"
done
