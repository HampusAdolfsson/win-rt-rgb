import { IProfile, IProfileCategory } from './Profile';
import { WebsocketService } from '../WebsocketService';
import { BehaviorSubject } from 'rxjs';
import Path from 'path';
import { readFileSync, existsSync, mkdirSync, writeFileSync } from 'fs';

export class ProfilesService {
  private static readonly profilesSaveFile = Path.join(process.env.APPDATA || ".", "win-rt-rgb", "profiles2.json");
  private static readonly idSaveFile = Path.join(process.env.APPDATA || ".", "win-rt-rgb", "profiles_id2.json");

  public static get Instance() {
    if (!this.instance) {
      this.LoadAndInstantiate();
    }
    return this.instance!;
  }

  public static LoadAndInstantiate() {
    let profiles: IProfileCategory[] = [];
    let nextId = 0;
    if (existsSync(this.profilesSaveFile)) {
      profiles = JSON.parse(readFileSync(this.profilesSaveFile).toString());
      profiles.forEach(cat => {
        cat.profiles.forEach(prof => {
          prof.areas.forEach(area => {
            // @ts-ignore
            if (area.selector === "*") {
              area.selector = undefined;
            }
          });
        });
      });
    }
    if (existsSync(this.idSaveFile)) {
      nextId = JSON.parse(readFileSync(this.idSaveFile).toString()).nextId;
    }
    this.instance = new ProfilesService(profiles, nextId);
  }

  private static instance: ProfilesService | undefined;

  public readonly categories: BehaviorSubject<IProfileCategory[]>;
  public readonly activeProfiles = new BehaviorSubject<Map<number, number>>(new Map());

  private nextId: number;

  private constructor(initialCategories: IProfileCategory[], nextId: number) {
    this.categories = new BehaviorSubject(initialCategories);
    this.nextId = nextId;

    this.sendProfiles(initialCategories);
    WebsocketService.Instance.receivedMessage.subscribe(message => {
      if (message.subject === 'activeProfile') {
        const monitorIndex = message.contents["monitor"];
        let newMap = new Map(this.activeProfiles.value);
        if (message.contents["profile"] != undefined) {
          newMap.set(monitorIndex, message.contents["profile"]);
        } else {
          newMap.delete(monitorIndex);
        }
        this.activeProfiles.next(newMap);
      }
    });
  }

  public setProfiles(categories: IProfileCategory[]) {
    this.categories.next(categories);
    this.sendProfiles(categories);
    if (!existsSync(Path.dirname(ProfilesService.profilesSaveFile))) {
      mkdirSync(Path.dirname(ProfilesService.profilesSaveFile), { recursive: true });
    }
    writeFileSync(ProfilesService.profilesSaveFile, JSON.stringify(categories));
  }

  public createProfile(): IProfile {
    const profile: IProfile = {
      id: this.nextId,
      regex: '',
      priority: undefined,
      areas: [{
        selector: undefined,
        x: { px: 0 },
        y: { px: 0 },
        width: { percentage: 100 },
        height: { percentage: 100 },
      }]
    };
    this.nextId += 1;

    writeFileSync(ProfilesService.idSaveFile, JSON.stringify({ nextId: this.nextId }));
    return profile;
  }

  private sendProfiles(categories: IProfileCategory[]) {
    const flattenedProfiles = categories.flatMap(category => {
      const profiles: IProfile[] = JSON.parse(JSON.stringify(category.profiles));
      profiles.forEach(profile => {
        if (profile.priority === undefined) {
          profile.priority = category.priority;
        }
      });
      return profiles;
    });
    WebsocketService.Instance.sendMessage('profiles', flattenedProfiles);
  }

}