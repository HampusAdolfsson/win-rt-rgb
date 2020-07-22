using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Sockets;
using System.Runtime.InteropServices.WindowsRuntime;
using Windows.ApplicationModel;
using Windows.ApplicationModel.Activation;
using Windows.ApplicationModel.Background;
using Windows.Foundation;
using Windows.Foundation.Collections;
using Windows.UI.Notifications;
using Windows.UI.Notifications.Management;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Controls.Primitives;
using Windows.UI.Xaml.Data;
using Windows.UI.Xaml.Input;
using Windows.UI.Xaml.Media;
using Windows.UI.Xaml.Navigation;

namespace DiscordNotificationListener
{
    /// <summary>
    /// Provides application-specific behavior to supplement the default Application class.
    /// </summary>
    sealed partial class App : Application
    {
        private List<uint> acknowledgedNotificationIds = new List<uint>();
        /// <summary>
        /// Initializes the singleton application object.  This is the first line of authored code
        /// executed, and as such is the logical equivalent of main() or WinMain().
        /// </summary>
        public App()
        {
            this.InitializeComponent();
            this.Suspending += OnSuspending;
            init();
        }

        public async void init() {
            UserNotificationListener listener = UserNotificationListener.Current;
            UserNotificationListenerAccessStatus accessStatus = await listener.RequestAccessAsync();

            switch (accessStatus) {
                case UserNotificationListenerAccessStatus.Allowed:

                    if (!BackgroundTaskRegistration.AllTasks.Any(i => i.Value.Name.Equals("UserNotificationChanged"))) {
                        var builder = new BackgroundTaskBuilder() {
                            Name = "UserNotificationChanged"
                        };

                        builder.SetTrigger(new UserNotificationChangedTrigger(NotificationKinds.Toast));

                        builder.Register();
                    }
                    break;

                case UserNotificationListenerAccessStatus.Denied:
                    break;
                case UserNotificationListenerAccessStatus.Unspecified:
                    break;
            }
        }

        /// <summary>
        /// Invoked when the application is launched normally by the end user.  Other entry points
        /// will be used such as when the application is launched to open a specific file.
        /// </summary>
        /// <param name="e">Details about the launch request and process.</param>
        protected override void OnLaunched(LaunchActivatedEventArgs e)
        {
            Frame rootFrame = Window.Current.Content as Frame;

            // Do not repeat app initialization when the Window already has content,
            // just ensure that the window is active
            if (rootFrame == null)
            {
                // Create a Frame to act as the navigation context and navigate to the first page
                rootFrame = new Frame();

                rootFrame.NavigationFailed += OnNavigationFailed;

                if (e.PreviousExecutionState == ApplicationExecutionState.Terminated)
                {
                    //TODO: Load state from previously suspended application
                }

                // Place the frame in the current Window
                Window.Current.Content = rootFrame;
            }

            if (e.PrelaunchActivated == false)
            {
                if (rootFrame.Content == null)
                {
                    // When the navigation stack isn't restored navigate to the first page,
                    // configuring the new page by passing required information as a navigation
                    // parameter
                    rootFrame.Navigate(typeof(MainPage), e.Arguments);
                }
                // Ensure the current window is active
                Window.Current.Activate();
            }
        }

        /// <summary>
        /// Invoked when Navigation to a certain page fails
        /// </summary>
        /// <param name="sender">The Frame which failed navigation</param>
        /// <param name="e">Details about the navigation failure</param>
        void OnNavigationFailed(object sender, NavigationFailedEventArgs e)
        {
            throw new Exception("Failed to load Page " + e.SourcePageType.FullName);
        }

        /// <summary>
        /// Invoked when application execution is being suspended.  Application state is saved
        /// without knowing whether the application will be terminated or resumed with the contents
        /// of memory still intact.
        /// </summary>
        /// <param name="sender">The source of the suspend request.</param>
        /// <param name="e">Details about the suspend request.</param>
        private void OnSuspending(object sender, SuspendingEventArgs e)
        {
            var deferral = e.SuspendingOperation.GetDeferral();
            //TODO: Save application state and stop any background activity
            deferral.Complete();
        }
        protected override async void OnBackgroundActivated(BackgroundActivatedEventArgs args) {
            var deferral = args.TaskInstance.GetDeferral();

            switch (args.TaskInstance.Task.Name) {
                case "UserNotificationChanged":
                    IReadOnlyList<UserNotification> userNotifications = await UserNotificationListener.Current.GetNotificationsAsync(NotificationKinds.Toast);
                    var newNotifs = userNotifications.Where(notif => !acknowledgedNotificationIds.Any(id => notif.Id == id));
                    if (newNotifs.Any(newNotif => newNotif.AppInfo.AppUserModelId == "com.squirrel.Discord.Discord")) {
                        TcpClient client = new TcpClient("192.168.1.6", 8844);
                        Byte[] payload = new Byte[] { 0x46, 0x50, 0x03, 0x01, 0, 0, 0x00, 0x00, 0x00, 0x00, 0x3b, 0x9a, 0xca, 0x00, 2, 0, 0, 0xff, 0x11, 0x11, 0xff };
                        client.GetStream().Write(payload, 0, payload.Length);
                        int res = client.GetStream().Read(payload, 0, 1);
                        client.Close();
                    }
                    acknowledgedNotificationIds.AddRange(newNotifs.Select(notif => notif.Id));

                    break;
            }

            deferral.Complete();
        }

    }
}
