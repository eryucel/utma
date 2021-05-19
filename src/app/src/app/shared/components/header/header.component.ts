import {Component, OnInit} from '@angular/core';
import {User, UserService} from "../../../core";

@Component({
  selector: 'app-header',
  templateUrl: './header.component.html',
  styleUrls: ['./header.component.css']
})
export class HeaderComponent implements OnInit {

  currentUser!: User;

  constructor(
    private userService: UserService) {
  }

  ngOnInit(): void {
    this.userService.currentUser.subscribe(
      (userData) => {
        this.currentUser = userData;
      }
    );
  }

}
