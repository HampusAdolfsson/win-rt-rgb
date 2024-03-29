import React from 'react';
import { render } from 'react-dom';

import 'fontsource-roboto';
import { CssBaseline, createMuiTheme, AppBar, Tabs, Tab, Typography } from '@material-ui/core';
import { ThemeProvider } from '@material-ui/core';
import { DevicesScene } from './devices/DevicesScene';
import { AboutScene } from './AboutScene';

import './styles/Global.css';
import { ProfilesScene } from './profiles/ProfilesScene';
import Alert from '@material-ui/lab/Alert';
import { WebsocketService } from './WebsocketService';
import { ProfilesService } from './profiles/ProfilesService';
import { Cast, Crop, Info } from '@material-ui/icons';

const mainElement = document.createElement('div');
mainElement.setAttribute('id', 'root');
document.body.appendChild(mainElement);

let theme = createMuiTheme({
  palette: {
    type: 'dark',
    primary: {
      light: '#ffe54c',
      main: '#ffb300',
      dark: '#c68400',
      contrastText: '#000',
    },
    secondary: {
      light: '#df78ef',
      main: '#ab47bc',
      dark: '#790e8b',
      contrastText: '#fff',
    },
  }
});

interface State {
  visibleScene: number;
  showBackendError: boolean;
  tabValue: number;
}

class App extends React.Component<{}, State> {

  constructor(props: {}) {
    super(props);
    this.state = {
      visibleScene: 0,
      showBackendError: false,
      tabValue: 0,
    };
    WebsocketService.Instance.connected.then(connected => {
      if (!connected) {
        this.setState({
          showBackendError: true,
        });
      }
    });
  }

  setScene(i: number) {
    this.setState({
      visibleScene: i,
    });
  }

  componentDidMount() {
    ProfilesService.LoadAndInstantiate();
  }

  render() {
    const value = this.state.tabValue;
    return (
      <>
        <ThemeProvider theme={theme} >
          <CssBaseline />
          <AppBar position="sticky" color="default">
            <Typography variant="h4" color="textSecondary" style={{position: "fixed", top: 15, left: 20 }}>win-rt-rgb</Typography>
            <Tabs value={value} onChange={(_, val) => { this.setState({tabValue: val}); }} centered
                  indicatorColor="primary"
                  textColor="primary" >
              <Tab icon={<Cast/>} label="Devices" />
              <Tab icon={<Crop/>} label="Profiles" />
              <Tab icon={<Info/>} label="About" />
            </Tabs>
          </AppBar>
          <div className="scene">
            {this.state.showBackendError && <Alert severity="error" variant="filled" style={{ marginBottom: 20 }}>
              Unable to connect to backend. Try restarting the application.</Alert>}
            {value == 0 && <DevicesScene/>}
            {value == 1 && <ProfilesScene/>}
            {value == 2 && <AboutScene/>}
          </div>
        </ThemeProvider>
      </>
    );
  }
}

render(<App />, mainElement);
