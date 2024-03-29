import { Button, createStyles, Dialog, DialogActions, DialogContent, DialogTitle, Icon, IconButton, Link, makeStyles, Switch, TableCell, Theme, Typography } from '@material-ui/core';
import { Delete, PowerSettingsNew, Settings, WbIncandescent } from '@material-ui/icons';
import React, { useEffect, useState } from 'react';
import { DeviceSettings } from './DeviceSettings';
import { IDeviceSpecification, DeviceTypes } from './DeviceSpecification';

const useStyles = makeStyles((theme: Theme) =>
  createStyles({
    deleteButton: {
      color: "#ff5555"
    },
  }),
);

interface Props {
  device: IDeviceSpecification;
  enabled: boolean;
  onDeviceDeleted: () => void;
  onDeviceChanged: (device: IDeviceSpecification) => void;
  onDeviceEnabledChanged: (enabled: boolean) => void;
}

enum WledPowerStatus {
  ON, OFF, UNREACHABLE
}

export function DeviceEntry(props: Props) {
  const classes = useStyles();
  const [enabled, setEnabled] = useState(props.enabled);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [powerState, setPowerState] = useState(WledPowerStatus.UNREACHABLE);

  if (props.device.type === DeviceTypes.WLED) {
    useEffect(() => {
      const interval = setInterval(async() => {
        try {
          const res = await fetch(`http://${props.device.wledData?.ipAddress}/json`);
          if (res.status !== 200) {
            setPowerState(WledPowerStatus.UNREACHABLE);
            return;
          }
          const data = await res.json();
          setPowerState(data["state"]["on"] ? WledPowerStatus.ON : WledPowerStatus.OFF);
        } catch (e) {
          setPowerState(WledPowerStatus.UNREACHABLE);
        }
      }, 1000);
      return () => {
        clearInterval(interval);
      };
    });
  }

  return <React.Fragment>
            <TableCell component="th" scope="row" >
              <Switch checked={enabled} onChange={(_, v) => { setEnabled(v); props.onDeviceEnabledChanged(v); }}/>
              <Typography variant="subtitle1" display="inline">{props.device.name}</Typography>
            </TableCell>
            <TableCell align="right" >{props.device.numberOfLeds} LEDs</TableCell>
            <TableCell align="right" >
              {props.device.type == 0 ?
                <>WLED - <Link target="_blank" href={"http://"+props.device.wledData?.ipAddress}>{props.device.wledData?.ipAddress}</Link></> :
                <>Qmk - {truncate(props.device.qmkData?.hardwareId || "", 13)} </>}
            </TableCell>
            <TableCell align="right" >
              {props.device.type == DeviceTypes.WLED && props.device.wledData?.ipAddress &&
                <IconButton color={powerState === WledPowerStatus.ON ? "primary" : "default"} disabled={powerState === WledPowerStatus.UNREACHABLE} onClick={() => {
                  const xmlHttp = new XMLHttpRequest();
                  xmlHttp.open( "GET", `http://${props.device.wledData?.ipAddress}/win&T=2`, true);
                  xmlHttp.send( null );
                }}><PowerSettingsNew/></IconButton>}
              <IconButton onClick={() => {setDialogOpen(true);}}>
                <Settings/>
              </IconButton>
              <IconButton onClick={props.onDeviceDeleted}>
                <Delete className={classes.deleteButton}/>
              </IconButton>
            </TableCell>
            <DeviceSettings device={props.device} open={dialogOpen}
              onClosed={() => {setDialogOpen(false);}}
              onDeviceChanged={(device) => {setDialogOpen(false); props.onDeviceChanged(device);}}
            />
         </React.Fragment>
}

function truncate(str: string, maxLength: number): string {
  if (str.length < maxLength) return str;
  return str.substring(0, maxLength - 1) + "…";
}