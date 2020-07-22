using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Windows.UI.Notifications.Management;
using Windows.UI.Notifications;
using System.Net.Sockets;
using Windows.ApplicationModel.Background;

namespace DiscordNotificationListener
{
    class NotificationListener
    {
        public async void Initialize() {
            // Get the listener
            UserNotificationListener listener = UserNotificationListener.Current;

            // And request access to the user's notifications (must be called from UI thread)
            UserNotificationListenerAccessStatus accessStatus = await listener.RequestAccessAsync();

            switch (accessStatus) {
                // This means the user has granted access.
                case UserNotificationListenerAccessStatus.Allowed:

                    // Subscribe to foreground event
                    listener.NotificationChanged += Listener_NotificationChanged;
                    //if (!BackgroundTaskRegistration.AllTasks.Any(i => i.Value.Name.Equals("UserNotificationChanged"))) {
                    //    var builder = new BackgroundTaskBuilder() {
                    //        Name = "UserNotificationChanged"
                    //    };

                    //    builder.SetTrigger(new UserNotificationChangedTrigger(NotificationKinds.Toast));

                    //    builder.Register();
                    //}
                    break;

                // This means the user has denied access.
                // Any further calls to RequestAccessAsync will instantly
                // return Denied. The user must go to the Windows settings
                // and manually allow access.
                case UserNotificationListenerAccessStatus.Denied:

                    // Show UI explaining that listener features will not
                    // work until user allows access.
                    break;

                // This means the user closed the prompt without
                // selecting either allow or deny. Further calls to
                // RequestAccessAsync will show the dialog again.
                case UserNotificationListenerAccessStatus.Unspecified:

                    // Show UI that allows the user to bring up the prompt again
                    break;
            }

        }
        private async void Listener_NotificationChanged(UserNotificationListener sender, UserNotificationChangedEventArgs args) {
            // Your code for handling the notification
            //Console.WriteLine(args.UserNotificationId);
            IReadOnlyList<UserNotification> userNotifications = await sender.GetNotificationsAsync(NotificationKinds.Toast);
            try {
                var newNotif = userNotifications.First(notif => notif.Id == args.UserNotificationId);
                if (newNotif != null && newNotif.AppInfo.AppUserModelId == "com.squirrel.Discord.Discord") {
                    TcpClient client = new TcpClient("192.168.1.6", 8844);
                    Byte[] payload = new Byte[]{ 0x46, 0x50, 0x03, 0x01, 0, 0, 0x00, 0x00, 0x00, 0x00, 0x3b, 0x9a, 0xca, 0x00, 2, 0,0,0xff, 0x11,0x11,0xff };
                    client.GetStream().Write(payload, 0, payload.Length);
                    int res = client.GetStream().Read(payload, 0, 1);
                    client.Close();
                }
            } catch (Exception e) { }
        }
    }
}
