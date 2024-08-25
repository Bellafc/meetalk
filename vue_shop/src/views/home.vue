<template>
  <el-card>
    <div class="header">
      <img src="../assets/home.png" style="width: 18px; height: 18px" />
      <span> Meetalk-Your Personal Meeting Helper </span>
    </div>

    <div
      style="width: 100%;display: flex;flex-direction: row;align-content: center;justify-content: center;flex-wrap: wrap;"
    >
      <div :style="'width:'+cards.card1Width+'%;'" ref="card1">
        <div class="card1">
          <div
            class="upload-container"
            style="display: flex; justify-content: space-between;"
          >
            <el-upload
              class="upload-el"
              ref="uploadAudio"
              :auto-upload="true"
              action="http://localhost:8080/upload"
              :on-change="handleChange"
              :style="{
                backgroundColor: '#4578b8',
                borderColor: '#4578b8',
                color: '#ffffff',
                width: 'calc(38% - 1px)'
              }"
            >
              <div
                class="upload-btn"
                style="
                  font-family: 'Arial Black';
                  font-size: 14px;
                  text-align: center;
                "
              >
                ⏫Upload audio / transcript
              </div>
            </el-upload>

            <el-upload
              class="upload-el"
              ref="uploadFiles"
              :auto-upload="true"
              action="http://localhost:8080/upload_doc"
              :on-change="handleChange_doc"
              :style="{
                backgroundColor: '#4578b8',
                borderColor: '#4578b8',
                color: '#ffffff',
                width: 'calc(38% - 1px)'
              }"
            >
              <div
                class="upload-btn"
                style="
                  font-family: 'Arial Black';
                  font-size: 14px;
                  text-align: center;
                "
              >
                ⏫Upload sample meeting minutes
              </div>
            </el-upload>

            <el-button
              class="upload-el"
              :auto-upload="true"
              @click="suggestchapter"
              :style="{
                backgroundColor: '#4578b8',
                borderColor: '#4578b8',
                color: '#ffffff',
                width: 'calc(24% - 1px)'
              }"
            >
              <div
                class="upload-btn1"
                style="
                  font-family: 'Arial Black';
                  font-size: 15px;
                  text-align: left;
                  margin-top: 4px;
                  margin-right: 5px;
                "
              >Suggest
              </div>
            </el-button>
          </div>
          <div class="card-el-one" style="margin-bottom: 10px;">
            <div class="card-main-one">
              <div class="card-header" style="font-family: 'Arial Black'; font-size: 25px;">
                <span>Chapters and styles</span>
              </div>
              <div class="card-container">
                <div
                  class="chapter"
                  v-for="(chaper, index) in chapterList"
                  :key="index"
                >
                  <el-card>
                    <div class="chapter-item">
                      <el-input
                        style="margin-right:8px;"
                        placeholder="Input chapter"
                        v-model="chaper.chapter"
                      ></el-input>
                      <el-button
                        icon="el-icon-circle-plus-outline"
                        @click="addSection(index)"
                        :style="{
                          backgroundColor: '#2a3f56',
                          borderColor: '#2a3f56',
                          color: '#ffffff'
                        }"
                      ></el-button>
                    </div>
                    <div
                      class="section"
                      v-for="(section, idx) in chaper.sectionList"
                      :key="`${index}_${idx}`"
                    >
                      <div class="section-item">
                        <el-input
                          style="margin-right: 8px; width: 50%"
                          placeholder="Input section"
                          v-model="section.section"
                          :style="{
                            backgroundColor: '#d9dee4',
                            borderColor: '#d9dee4',
                            color: '#ffffff'
                          }"
                        ></el-input>
                      <el-select
                        placeholder="Select writing Style"
                        :value="section.writingStyle"
                        :style="{
                          backgroundColor: '#d9dee4',
                          borderColor: '#d9dee4',
                          color: '#ffffff'
                        }"
                        @change="
                          (val) => {
                            handleSelect(val, index, idx);
                          }
                        "
                      >
                        <el-option
                          v-for="item in writingStyleList"
                          :key="item.value"
                          :label="item.label"
                          :value="item.value"
                          :style="{
                            backgroundColor: '#d9dee4',
                            borderColor: '#d9dee4',
                            color: '#ffffff'
                          }"
                        >
                        </el-option>
                      </el-select>
                    </div>
                    </div>
                  </el-card>
                </div>
              <el-button
                icon="el-icon-circle-plus-outline"
                style="margin: 2px 0; width: 100%"
                @click="addChapter"
                :style="{
                  backgroundColor: '#2a3f56',
                  borderColor: '#4578b8',
                  color: '#ffffff'
                }"
              >
                Add Chapter
              </el-button>
              <el-button
                style="margin-left: 0; width: 100%"
                @click="handleSubmit"
                :style="{
                  backgroundColor: '#2a3f56',
                  borderColor: '#4578b8',
                  color: '#ffffff'
                }"
              >
                Submit
              </el-button>
              </div>
          </div>
            </div>
          <div class="card-el-one">
            <div class="card-main-two">
              <div class="card-header" style="font-family: 'Arial Black'; font-size: 25px;">
                <span>Update Writing Styles</span>
              </div>
              <div class="card-container-two">
                <div
                  id="updatedchapterlist"
                  contenteditable="true"
                  style="word-wrap: break-word; white-space: pre-wrap;font-size: 18px;"
                >
                  <pre id="newwritingstyle" style="white-space: pre-wrap;font-size: 14px;"></pre>
                </div>
                <el-button
                  id="update"
                  @click="update"
                  :style="{
                    backgroundColor: '#2a3f56',
                    borderColor: '#2a3f56',
                    color: '#ffffff'
                  }"
                >
                  Update
                </el-button>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div class="resize" :style="'width:'+cards.resizeLine1+'%;'" ref="resizeLine1Ref"></div>
      <div :style="'width:'+cards.cardWidth+'%;'" ref="card">
        <div class="card">
          <div class="card-el" style="margin-right: 5px;height:100%;">
            <div class="card-header" style="font-family: 'Arial Black'; font-size: 25px;">
              <span>Chapter allocation 📖</span>
            </div>
            <div style="margin-left: 60px;" ref="secondBtnBox">
              <el-button
                id="pause"
                @click="pause"
                :style="{
                  backgroundColor: '#2a3f56',
                  borderColor: '#2a3f56',
                  color: '#ffffff'
                }"
              >
                Pause
              </el-button>
              <el-button
                id="chap"
                @click="chapmodify"
                :style="{
                  backgroundColor: '#2a3f56',
                  borderColor: '#4578b8',
                  color: '#ffffff'
                }"
              >
                Upload Modification
              </el-button>
              <el-button
                id="goon"
                @click="goon"
                :style="{
                  backgroundColor: '#2a3f56',
                  borderColor: '#4578b8',
                  color: '#ffffff'
                }"
              >
                Continue
              </el-button>
              <el-button
                id="newwritingstyle"
                @click="newwritingstyle"
                :style="{
                  backgroundColor: '#2a3f56',
                  borderColor: '#4578b8',
                  color: '#ffffff'
                }"
              >
                Add New Writing Style
              </el-button>
              <el-button
                id="written"
                @click="written2"
                :style="{
                  backgroundColor: '#2a3f56',
                  borderColor: '#4578b8',
                  color: '#ffffff'
                }"
              >
                Write
              </el-button>
            </div>

            <div
              class="card-main"
              :style="'margin-top: 5px;height:'+ (800 - 80 - secondBoxBtnHeight) + 'px; !important'"
            >
              <div
                id="chapsDisplay"
                contenteditable="true"
                style="
                  word-wrap: break-word;
                  white-space: pre-wrap;
                  font-size: 14px;
                  margin-top: 5px;
                  height: 100%;
                  overflow-y: auto;
                "
              >
                <pre style="white-space: pre-wrap;font-size: 14px;"></pre>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div class="resize" :style="'width:'+cards.resizeLine2+'%;'" ref="resizeLine2Ref"></div>
      <div :style="'width:'+cards.card2Width+'%;'" ref="card2">
        <div class="card2">
          <div class="card-el">
            <div class="card-header" style="font-family: 'Arial Black'; font-size: 25px;">
              <span>Meeting Minutes ✏️</span>
            </div>
            <div style="margin-left: 20px;">
              <el-button
                id="writtenmodify"
                @click="writtenmodify"
                :style="{
                  backgroundColor: '#2a3f56',
                  borderColor: '#4578b8',
                  color: '#ffffff'
                }"
              >
                Upload Writing Modification
              </el-button>
            </div>
            <div class="card-main" contenteditable="true" style="margin-top: 5px;">
              <p></p>
              <pre
                id="written2"
                style="
                  word-wrap: break-word;
                  white-space: pre-wrap;
                  font-size: 14px;
                "
              >
              </pre>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="card-el-three" style="margin-bottom: 5px; margin-top: 5px;">
      <div class="card-header" style="display: flex; align-items: center;">
        <span
          style="margin-right: auto; font-family: 'Arial Black'; font-size: 25px;"
          >Chapter Allocation Manager</span
        >
        <el-button
          id="allocationdata"
          @click="allocationdata"
          :style="{
            backgroundColor: '#2a3f56',
            borderColor: '#2a3f56',
            color: '#ffffff'
          }"
        >
          Chapter Allocation Data
        </el-button>
        <el-button
          type="primary"
          @click="addRow"
          :style="{
            backgroundColor: '#2a3f56',
            borderColor: '#2a3f56',
            color: '#ffffff'
          }"
        >
          Add Rows
        </el-button>
        <el-button
          type="success"
          @click="saveData"
          :style="{
            backgroundColor: '#2a3f56',
            borderColor: '#2a3f56',
            color: '#ffffff'
          }"
        >
          Save Data
        </el-button>
      </div>
      <div class="card-main-three" style="margin-right: 10px; width: 98%;">
        <div class="card-container-three">
          <el-table :data="allocationdataList" border stripe style="width: 100%">
            <el-table-column prop="content" label="content" width="1000">
              <template slot-scope="scope">
                <el-input
                  v-model="scope.row.current_content"
                  type="textarea"
                  :autosize="{ minRows: 2, maxRows: 6 }"
                  style="font-size: 12pt; font-family:'Times New Roman';"
                ></el-input>
              </template>
            </el-table-column>
            <el-table-column prop="label a" label="label a">
              <template slot-scope="scope">
                <el-input
                  v-model="scope.row.label1"
                  type="textarea"
                  :autosize="{ minRows: 2, maxRows: 6 }"
                  style="font-size: 14pt; font-family:'Times New Roman';font-weight:bold;"
                ></el-input>
              </template>
            </el-table-column>
            <el-table-column prop="label b" label="label b">
              <template slot-scope="scope">
                <el-input
                  v-model="scope.row.label2"
                  type="textarea"
                  :autosize="{ minRows: 2, maxRows: 6 }"
                  style="font-size: 14pt; font-family: 'Times New Roman';font-weight:bold;"
                ></el-input>
              </template>
            </el-table-column>
          </el-table>
        </div>
      </div>
    </div>

    <div class="card-el-four" style="margin-bottom: 5px; margin-top: 5px;">
      <div class="card-header" style="display: flex; align-items: center;">
        <span style="margin-right: auto; font-family: 'Arial Black'; font-size: 25px;"
          >Writing Style Manager</span
        >
        <el-button
          id="wsdata"
          @click="wsdata"
          :style="{
            backgroundColor: '#2a3f56',
            borderColor: '#2a3f56',
            color: '#ffffff'
          }"
        >
          Get Writing Style Data
        </el-button>
        <el-button
          type="primary"
          @click="addRowws"
          :style="{
            backgroundColor: '#2a3f56',
            borderColor: '#2a3f56',
            color: '#ffffff'
          }"
        >
          Add Rows
        </el-button>
        <el-button
          type="success"
          @click="saveDataws"
          :style="{
            backgroundColor: '#2a3f56',
            borderColor: '#2a3f56',
            color: '#ffffff'
          }"
        >
          Save Data
        </el-button>
      </div>
      <div class="card-main-four" style="margin-right: 10px; width: 98%;">
        <div class="card-container-four">
          <div class="new-content">
            <el-table :data="writingstyledataList" border stripe style="width: 100%">
              <el-table-column prop="tag" label="tag" width="100">
                <template slot-scope="scope">
                  <el-input
                    v-model="scope.row.tag"
                    type="textarea"
                    :autosize="{ minRows: 1, maxRows: 6 }"
                    class="custom-input"
                  ></el-input>
                </template>
              </el-table-column>
              <el-table-column prop="input" label="input">
                <template slot-scope="scope">
                  <el-input
                    v-model="scope.row.input"
                    type="textarea"
                    :autosize="{ minRows: 1, maxRows: 6 }"
                    style="font-size: 10pt; font-family:'Times New Roman';"
                  ></el-input>
                </template>
              </el-table-column>
              <el-table-column prop="participant" label="participant">
                <template slot-scope="scope">
                  <el-input
                    v-model="scope.row.participant"
                    type="textarea"
                    :autosize="{ minRows: 1, maxRows: 6 }"
                    style="font-size: 10pt; font-family: 'Times New Roman';"
                  ></el-input>
                </template>
              </el-table-column>
              <el-table-column prop="writing_goal" label="writing_goal">
                <template slot-scope="scope">
                  <el-input
                    v-model="scope.row.writing_goal"
                    type="textarea"
                    :autosize="{ minRows: 1, maxRows: 6 }"
                    style="font-size: 10pt; font-family: 'Times New Roman';"
                  ></el-input>
                </template>
              </el-table-column>
              <el-table-column prop="writing_format" label="writing_format">
                <template slot-scope="scope">
                  <el-input
                    v-model="scope.row.writing_format"
                    type="textarea"
                    :autosize="{ minRows: 1, maxRows: 6 }"
                    style="font-size: 10pt; font-family: 'Times New Roman';"
                  ></el-input>
                </template>
              </el-table-column>
              <el-table-column prop="your_role" label="your_role">
                <template slot-scope="scope">
                  <el-input
                    v-model="scope.row.your_role"
                    type="textarea"
                    :autosize="{ minRows: 1, maxRows: 6 }"
                    style="font-size: 10pt; font-family: 'Times New Roman';"
                  ></el-input>
                </template>
              </el-table-column>
              <el-table-column prop="analytical_thinking" label="analytical_thinking">
                <template slot-scope="scope">
                  <el-input
                    v-model="scope.row.analytical_thinking"
                    type="textarea"
                    :autosize="{ minRows: 1, maxRows: 6 }"
                    style="font-size: 10pt; font-family: 'Times New Roman';"
                  ></el-input>
                </template>
              </el-table-column>
              <el-table-column prop="clout" label="clout">
                <template slot-scope="scope">
                  <el-input
                    v-model="scope.row.clout"
                    type="textarea"
                    :autosize="{ minRows: 1, maxRows: 6 }"
                    style="font-size: 10pt; font-family: 'Times New Roman';"
                  ></el-input>
                </template>
              </el-table-column>
              <el-table-column prop="authentic" label="authentic">
                <template slot-scope="scope">
                  <el-input
                    v-model="scope.row.authentic"
                    type="textarea"
                    :autosize="{ minRows: 1, maxRows: 6 }"
                    style="font-size: 10pt; font-family: 'Times New Roman';"
                  ></el-input>
                </template>
              </el-table-column>
              <el-table-column prop="emotional_tone" label="emotional_tone" width="100">
                <template slot-scope="scope">
                  <el-input
                    v-model="scope.row.emotional_tone"
                    :autosize="{ minRows: 1, maxRows: 6 }"
                    style="font-size: 10pt; font-family: 'Times New Roman';"
                  ></el-input>
                </template>
              </el-table-column>
              <el-table-column prop="language" label="language" width="100">
                <template slot-scope="scope">
                  <el-input
                    v-model="scope.row.language"
                    type="textarea"
                    :autosize="{ minRows: 1, maxRows: 6 }"
                    style="font-size: 10pt; font-family: 'Times New Roman';"
                  ></el-input>
                </template>
              </el-table-column>
              <el-table-column prop="difference" label="difference" width="100">
                <template slot-scope="scope">
                  <el-input
                    v-model="scope.row.difference"
                    type="textarea"
                    :autosize="{ minRows: 1, maxRows: 6 }"
                    style="font-size: 10pt; font-family: 'Times New Roman';"
                  ></el-input>
                </template>
              </el-table-column>
            </el-table>
          </div>
        </div>
      </div>
    </div>
  </el-card>
</template>

<script>
import axios from "axios";

export default {
  data() {
    return {
      file: null,
      message: "",
      transposedData: [],
      chaps: "",
      chapterList: [
        {
          chapter: "",
          sectionList: [{ section: "", writingStyle: "" }],
        },
      ],
      writingStyleList: [
        {
          label: "default",
          value: "default",
        },
        {
          label: "question",
          value: "question",
        },
        {
          label: "participant",
          value: "participant",
        },
        {
          label: "finance",
          value: "finance",
        },
        {
          label: "legal suggestion",
          value: "legal suggestion",
        },
        {
          label: "event",
          value: "event",
        },
      ],
      resultList: [],
      allocationdataList: [],
      writingstyledataList: [],
      suggestchapterList:[],
      secondBoxBtnHeight: 0,
      clientWidth: 0,
      cards: {
        card1Width: 0,
        cardWidth: 0,
        card2Width: 0,
        resizeLine1: 0,
        resizeLine2: 0,
      },
      isLoding:false
    };
  },
  mounted() {
    setTimeout(() => {
      this.secondBoxBtnHeight = this.$refs.secondBtnBox.offsetHeight;
    }, 300);
    this.clientWidth = document.body.clientWidth;
    this.initCarWidth();
    this.handleResize();
    window.onresize = () => {
      return (() => {
        this.secondBoxBtnHeight = this.$refs.secondBtnBox.offsetHeight;
        this.clientWidth = document.body.clientWidth;
        this.initCarWidth();
      })();
    };
  },

  methods: {
    loadingApi() {
      this.isLoading = this.$loading({
        lock: true,
        text: 'Loading',
        spinner: 'el-icon-loading',
        background: 'rgba(0, 0, 0, 0.7)'
      });
      setTimeout(() => {
        this.isLoading.close();
      }, 2000);
    },
    suggestchapter() {
      console.log('点击了suggest,')
      // 点击了登录如何发送http请求？axios
      const params = {}
      this.loadingApi();
      const res = axios.post('http://localhost:8080/suggestchapter', params)
      // 异步操作成功时，执行的回调函数
      res.then(response => {
        console.log(response);
        //上
        this.chapterList = response.data.data.chapterList;
        //中
        this.allocationdataList = response.data.data.allocationdataList;
        //下
        this.writingstyledataList = response.data.data.writingstyledataList;
        this.isLoading.close();
        console.log(this.chapterList,"chapterList")
      });
      // 异步操作失败时，执行的回调函数
      res.catch(error => {
        this.$message.error("请求失败!")
      });
    },
    initCarWidth() {
      if (this.clientWidth < 1200) {
        this.$nextTick(() => {
          this.cards.card1Width = 100;
          this.cards.cardWidth = 100;
          this.cards.card2Width = 100;
          this.cards.resizeLine1 = 0;
          this.cards.resizeLine2 = 0;
        });
      } else {
        this.$nextTick(() => {
          this.cards.card1Width = 25.4;
          this.cards.cardWidth = 44;
          this.cards.card2Width = 30;
          this.cards.resizeLine1 = 0.3;
          this.cards.resizeLine2 = 0.3;
        });
      }
    },
    handleResize() {
      let resizeLine1Ref = this.$refs.resizeLine1Ref;
      let resizeLine2Ref = this.$refs.resizeLine2Ref;
      resizeLine1Ref.onmousedown = (e) => {
        document.onmousemove = (e) => {
          if (e.movementX < 0) {
            this.cards.card1Width -= 0.1;
            this.cards.cardWidth += 0.1;
          } else {
            this.cards.card1Width += 0.1;
            this.cards.cardWidth -= 0.1;
          }
          this.secondBoxBtnHeight = this.$refs.secondBtnBox.offsetHeight;
        };

        document.onmouseup = () => {
          document.onmousemove = null;
          document.onmouseup = null;
          resizeLine1Ref.releaseCapture && resizeLine1Ref.releaseCapture(); // 鼠标捕获释放
        };
        resizeLine1Ref.setCapture && resizeLine1Ref.setCapture(); // 启用鼠标捕获
        return false;
      };
      resizeLine2Ref.onmousedown = (e) => {
        document.onmousemove = (e) => {
          if (e.movementX < 0) {
            this.cards.cardWidth -= 0.1;
            this.cards.card2Width += 0.1;
          } else {
            this.cards.cardWidth += 0.1;
            this.cards.card2Width -= 0.1;
          }
          this.secondBoxBtnHeight = this.$refs.secondBtnBox.offsetHeight;
        };

        document.onmouseup = () => {
          document.onmousemove = null;
          document.onmouseup = null;
          resizeLine2Ref.releaseCapture && resizeLine2Ref.releaseCapture(); // 鼠标捕获释放
        };
        resizeLine2Ref.setCapture && resizeLine2Ref.setCapture(); // 启用鼠标捕获

        return false;
      };
    },
    allocationdata() {
      fetch("http://localhost:8080/api/allocationdata", {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      })
        .then((response) => response.json())
        .then((data) => {
          if (data) {
            this.allocationdataList = JSON.parse(data);
          } else {
            this.allocationdataList = [];
          }
          console.log(this.allocationdataList, "allocationdataList");
        })
        .catch((error) => {
          console.error("Error:", error);
        });
    },
    addRow() {
      this.allocationdataList.push({
        current_content: "",
        label1: "",
        label2: "",
      });
    },
    wsdata() {
      fetch("http://localhost:8080/api/wsdata", {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      })
        .then((response) => response.json())
        .then((data) => {
          if (data) {
            this.writingstyledataList = JSON.parse(data);
          } else {
            this.writingstyledataList = [];
          }
          console.log(this.writingstyledataList, "writingstyledataList");
        })
        .catch((error) => {
          console.error("Error:", error);
        });
    },
    addRowws() {
      this.writingstyledataList.push({
        tag: "",
        input: "",
        participant: "",
        writing_goal: "",
        writing_format: "",
        your_role: "",
        analytical_thinking: "",
        clout: "",
        authentic: "",
        emotional_tone: "",
        language: "",
      });
    },

    pause() {
      console.log("Pausing the process");

      fetch("http://localhost:8080/api/pause", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({}),
      })
        .then((response) => response.json())
        .then((data) => {
          console.log("Success:", data);
        })
        .catch((error) => {
          console.error("Error:", error);
        });
    },
    goon() {
      console.log("Continuing the process");

      fetch("http://localhost:8080/api/goon", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({}),
      })
        .then((response) => response.json())
        .then((data) => {
          console.log("Success:", data);
        })
        .catch((error) => {
          console.error("Error:", error);
        });
    },
    update() {
      console.log("UPDATE THE CHAPTER LIST");
      var updatedchapterlist = document.getElementById("updatedchapterlist");
      let updatedchapterliststr = updatedchapterlist.textContent;
      console.log("got", updatedchapterliststr);
      fetch("http://localhost:8080/api/updatechapterlist", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(updatedchapterliststr),
      })
        .then((response) => response.json())
        .then((data) => {
          console.log("Success:", data);
        })
        .catch((error) => {
          console.error("Error:", error);
        });
    },
    saveData() {
      console.log("Saving data...");
      axios
        .post("http://localhost:8080/api/save-data", this.allocationdataList)
        .then((response) => {
          console.log("Success:", response.data);
          this.$message.success("Data saved");
        })
        .catch((error) => {
          console.error("Error:", error);
          this.$message.error("Data save failed");
        });
    },
    saveDataws() {
      console.log("Saving data...");
      axios
        .post("http://localhost:8080/api/save-data-ws", this.writingstyledataList)
        .then((response) => {
          console.log("Success:", response.data);
          this.$message.success("Data saved");
        })
        .catch((error) => {
          console.error("Error:", error);
          this.$message.error("Data save failed");
        });
    },

    chapmodify() {
      console.log("chapter allocation modified");
      var chapsDisplay = document.getElementById("chapsDisplay");
      let chatmodify = chapsDisplay.textContent;
      console.log("wwww", chatmodify);
      fetch("http://localhost:8080/api/chapmodify", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(chatmodify),
      })
        .then((response) => response.json())
        .then((data) => {
          console.log("Success:", data);
        })
        .catch((error) => {
          console.error("Error:", error);
        });
    },
    writtenmodify() {
      console.log("writting modified");
      var writtenDisplay = document.getElementById("written2");
      let writtenmodify = writtenDisplay.textContent;
      console.log("wwww", writtenmodify);
      fetch("http://localhost:8080/api/writtenmodify", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(writtenmodify),
      })
        .then((response) => response.json())
        .then((data) => {
          console.log("Success:", data);
        })
        .catch((error) => {
          console.error("Error:", error);
        });
    },
    newwritingstyle() {
      console.log("showing which sections have no writing styles");
      fetch("http://localhost:8080/api/newwritingstyle", {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      })
        .then((response) => response.json())
        .then((data) => {
          console.log("Success:", data);
          document.getElementById("newwritingstyle").textContent = data;
        })
        .catch((error) => {
          console.error("Error:", error);
        });
    },
    written2() {
      console.log("wwww");
      fetch("http://localhost:8080/api/written", {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      })
        .then((response) => response.json())
        .then((data) => {
          console.log("Success:", data);
          document.getElementById("written2").textContent = data;
        })
        .catch((error) => {
          console.error("Error:", error);
        });
    },
    handleChange(file, fileList) {
      let formData = new FormData();
      console.log("eeeeeee", file);
      formData.append("file", file);

      fetch("http://localhost:8080/upload", {
        method: "POST",
        body: formData,
      })
        .then((response) => {
          if (!response.ok) {
            throw new Error("Network response was not ok");
          }
          return response.json();
        })
        .then((data) => {
          console.log(data.message);
        })
        .catch((error) => {
          console.error("There was an error uploading the file!", error);
        });
    },
    handleChange_doc(file, fileList) {
      let formData2 = new FormData();
      console.log("eeeeeee", file);
      formData2.append("file", file);

      fetch("http://localhost:8080/upload_doc", {
        method: "POST",
        body: formData2,
      })
        .then((response) => {
          if (!response.ok) {
            throw new Error("Network response was not ok");
          }
          return response.json();
          console.log("response", response.json);
        })
        .then((data) => {
          console.log(data.message);
          // 更新 extractedContent
          this.extractedContent = data.data;
          // 更新 sectionList
          this.updateSectionList(data.data);
        })
        .catch((error) => {
          console.error("There was an error uploading the file!", error);
        });
    },
    updateSectionList(content) {
      // 将 JSON 内容转换为 sectionList 的格式
      this.chaper.sectionList = Object.keys(content).map((key) => ({
        section: key,
        items: content[key],
      }));
    },
    addChapter() {
      this.chapterList.push({
        chapter: "",
        sectionList: [{ section: "", writingStyle: "" }],
      });
    },
    addSection(index) {
      this.chapterList[index].sectionList.push({
        section: "",
        writingStyle: "",
      });
    },
    handleSelect(val, index, idx) {
      this.chapterList[index].sectionList[idx].writingStyle = val;
    },
    handleSubmit() {
      this.resultList = [];
      this.chapterList.forEach((chapterItem, index) => {
        chapterItem.sectionList.forEach((item) => {
          this.resultList.push({
            chapter: chapterItem.chapter,
            section: item.section,
            writingStyle: item.writingStyle,
          });
        });
      });
      console.log("this.resultList", this.resultList);
      const data = { resultList: this.resultList };

      // 使用 EventSource 连接到 /stream 端点

      // 向 /api/submitchapter 发送 POST 请求
      fetch("http://localhost:8080/api/submitchapter", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      })
        .then((response) => {
          console.log("Submit chapter response:", response);
          const eventSource = new EventSource("http://localhost:8080/api/submitchapter");

          eventSource.onmessage = (event) => {
            // const newMessage = event.data;
            var data = JSON.parse(event.data);
            var chaps = data.Result;
            console.log("Chaps", chaps);
            var chapsDisplay = document.getElementById("chapsDisplay");
            chapsDisplay.textContent = chaps;
            var formattedChaps = chaps.replace(/\(.*?\)/g, (match) => {
              console.log("match", match);
              if (match.includes("Not Sure")) {
                return `<strong style="color:red">${match}</strong>`;
              } else {
                return `<strong>${match}</strong>`;
              }
            });
            chapsDisplay.innerHTML = formattedChaps;
            console.log("formatted", formattedChaps);
          };

          eventSource.onerror = () => {
            console.error("EventSource failed.");
            eventSource.close();
          };
        })
        .catch((error) => {
          console.error("Error submitting chapter:", error);
        });
    },
  },
};
</script>

<style scoped>
.custom-input .el-textarea__inner {
  font-size: 15pt;
  font-family: "Arial Narrow";
}

.el-table .el-table__cell {
  line-height: 1.5 !important;
}

.header {
  display: flex;
  width: 100%;
  padding: 20px;
  background: #d9dee4;
  box-sizing: border-box;
}

.header span {
  padding-left: 12px;
}

.resize {
  cursor: col-resize;
  background-color: #fff;
}

.resize:hover {
  background-color: #d9dee4;
}

.card {
  border-radius: 10px;
  /*width:762px;*/
  height: 800px;
  background: #4578b8;
  margin-top: 20px;
}

.card2 {
  border-radius: 10px;
  /*width:460px;*/
  height: 800px;
  background: #4578b8;
  margin-top: 20px;
}

.card1 {
  border-radius: 10px;
  /*width: 415px;*/
  height: 800px;
  margin-top: 20px;
  background: #fff;
}

.card-header {
  display: flex;
  width: 100%;
  padding: 10px 20px;
  height: 60px;
  color: white;
  box-sizing: border-box;
  font-size: 30px;
  justify-content: center;
}

.card-main {
  margin: 0 20px;
  background: #d9dee4;
  border-radius: 10px;
  box-sizing: border-box;
  padding: 24px 24px;
  overflow-y: auto;
  height: 680px;
}

.card-container {
  height: 340px;
  background: #d9dee4;
  border-radius: 10px;
  box-sizing: border-box;
  padding: 20px 12px;
  overflow-y: auto;
}

.card-container-two {
  height: 230px;
  background: #d9dee4;
  border-radius: 10px;
  box-sizing: border-box;
  padding: 20px 12px;
  overflow-y: auto;
}

.card-container-three {
  height: 400px;
  background: #d9dee4;
  border-radius: 10px;
  box-sizing: border-box;
  padding: 20px 12px;
  overflow-y: auto;
}

.card-container-four {
  /*width: 1590px;*/
  height: 380px;
  background: #d9dee4;
  border-radius: 10px;
  box-sizing: border-box;
  padding: 20px 12px;
  overflow-y: auto;
}

.card-main-one {
  height: 420px;
  margin: 0 10px;
  box-sizing: border-box;
  overflow-y: auto;
}

.card-main-two {
  height: 310px;
  margin: 0 10px;
  box-sizing: border-box;
  overflow-y: auto;
}

.card-main-three {
  height: 470px;
  margin: 0 20px;
  box-sizing: border-box;
  overflow-y: auto;
}

.card-main-four {
  height: 450px;
  margin: 0 20px;
  box-sizing: border-box;
  overflow-y: auto;
}

.card-main-five {
  height: 890px;
  width: 1630px;
  margin: 0 20px;
  box-sizing: border-box;
  overflow-y: auto;
}

.card-el {
  height: 100%;
  margin-left: 5px;
}

.card-el-one {
  background: #4578b8;
  border-radius: 10px;
}

.card-el-two {
  background: #4578b8;
  height: 500px;
  border-radius: 10px;
}

.card-el-three {
  background: #4578b8;
  height: 470px;
  border-radius: 10px;
}

.card-el-four {
  background: #4578b8;
  height: 450px;
  border-radius: 10px;
}

.upload-el {
  width: 100%;
  background: #4578b8;
  border-radius: 10px;
  box-sizing: border-box;
  height: 50px;
  color: #fff;
  text-align: center;
  margin-bottom: 10px;
}

.upload-btn {
  width: 100%;
  background: #4578b8;
  border-radius: 10px;
  box-sizing: border-box;
  height: 30px;
  color: #fff;
  font-size: 30px;
  margin-top: 5px;
}
.upload-btn1 {
  width: 100%;
  background: #4578b8;
  border-radius: 10px;
  box-sizing: border-box;
  height: 30px;
  color: #fff;
  font-size: 30px;
  margin-top: 12px;
}

*::-webkit-scrollbar {
  width: 4px;
  height: 4px;
  z-index: 10;
}

*::-webkit-scrollbar-thumb {
  border-radius: 10px;
  background-color: fade(#535353, 15%);
}

*::-webkit-scrollbar-track {
  box-shadow: inset 0 0 5px rgba(0, 0, 0, 0.1);
  background: lighten(#ededed, 10%);
  border-radius: 10px;
}

.chapter,
.section {
  margin-top: 4px;
}

.chapter-item,
.section-item {
  display: flex;
}

.chapter-item {
  /* width: 90%; */
  margin: 0 auto;
}

/* CSS 类用于换行 */
.wrap-text {
  white-space: pre-line;
}

/* .section-item {
  margin-left: 5%;
} */
</style>
