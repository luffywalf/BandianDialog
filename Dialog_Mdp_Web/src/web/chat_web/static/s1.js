// 基于准备好的dom，初始化echarts实例
var myChart = echarts.init(document.getElementById('kg_window'));
// 下面定义的const 但其实还是可以改变其值
const data2 = [{'name': '用户', 'id': '1','children': [{'name': '低压', 'id': '2', 'children': [{'name': '居民', 'id': '3', 'children': [{'name': '流程', 'id': '4', 'children': [{'name': '申请受理', 'id': '5', 'children': [{'name': '在受理您的报装申请后我们将安排客户经理与您联系', 'id': '17'}, {'name': '材料', 'id': '6', 'children': [{'name': '委托人', 'id': '11', 'children': [{'name': '委托人的身份证明', 'id': '12'}, {'name': '委托授权书', 'id': '13'}]}, {'name': '客户有效身份证明', 'id': '7', 'children': [{'name': '户口簿', 'id': '10'}, {'name': '护照', 'id': '9'}, {'name': '身份证', 'id': '8'}]}, {'name': '房产证或租赁协议复印件（出具产权人房产证复印件及同意报装证明）', 'id': '14'}, {'name': '私人宅基地建筑', 'id': '15', 'children': [{'name': '政府或规划部门的批复文件', 'id': '16'}]}]}]}]}]}, {'name': '非居民', 'id': '18', 'children': [{'name': '流程', 'id': '19', 'children': [{'name': '申请受理', 'id': '20', 'children': [{'name': '外部工程施工', 'id': '38', 'children': [{'name': '在受理您的报装申请后，我们将安排客户经理与您约定的时间进行现场勘查、收取报装资料，并根据典型设计开展外部工程施工。对涉及在居民小区内等公共区域施工的，贵方有协调业委会、物业或村委会等相关方责任，以便顺利完成施工', 'id': '39'}, {'name': '装接电表', 'id': '40', 'children': [{'name': '在竣工检验合格，签订《供用电合同》及相关协议后，我们将为您装表接电', 'id': '41'}]}]}, {'name': '材料', 'id': '21', 'children': [{'name': '企业法定代表人身份证复印件，或法定代表人签章的授权本单位人员的授权委托书及被授权经办人的身份证复印件', 'id': '23'}, {'name': '农村', 'id': '32', 'children': [{'name': '乡镇及以上政府或规划部门的批复文件', 'id': '33'}]}, {'name': '国家优惠价', 'id': '36', 'children': [{'name': '政府有关部门核发的资质证明和公益证明（学校、养老院、居委会）', 'id': '37'}]}, {'name': '客户用电报装申请书、客户用电设备信息登记表（动力、照明）', 'id': '22'}, {'name': '房产证复印件或租赁协议复印件（出具房产证复印件及同意报装证明）', 'id': '31'}, {'name': '新建建筑物', 'id': '34', 'children': [{'name': '立项、规划批复文件', 'id': '35'}]}, {'name': '有效营业执照复印件', 'id': '24', 'children': [{'name': '企业或工商客户', 'id': '25', 'children': [{'name': '营业执照复印件', 'id': '26'}]}, {'name': '军队', 'id': '29', 'children': [{'name': '团级及以上证明', 'id': '30'}]}, {'name': '机关事业单位', 'id': '27', 'children': [{'name': '事业法人证书或组织机构代码证复印件', 'id': '28'}]}]}]}]}]}]}]}, {'name': '高压', 'id': '42', 'children': [{'name': '流程', 'id': '43', 'children': [{'name': '申请受理', 'id': '44', 'children': [{'name': '方案答复', 'id': '74', 'children': [{'name': '在受理您的报装申请后， 我们将安排客户经理与您约定的时间进行现场勘查，出具供电方案并向您答复', 'id': '75'}, {'name': '外部工程施工', 'id': '76', 'children': [{'name': '装接电表', 'id': '78', 'children': [{'name': '在竣工检验合格， 签订《供用电合同》及相关协议后，我们将为您装表接电', 'id': '79'}]}, {'name': '请您自主选择具备相应资质的设计及施工单位进行设计施工。 工程竣工后，请及时报验，我们将安排竣工检验', 'id': '77'}]}]}, {'name': '材料', 'id': '45', 'children': [{'name': '主体证明', 'id': '46', 'children': [{'name': '企业法定代表人身份证复印件，或法定代表人签章的授权本单位人员的授权委托书及被授权经办的身份证复印件', 'id': '54'}, {'name': '有效营业执照复印件', 'id': '47', 'children': [{'name': '企业', 'id': '48', 'children': [{'name': '营业执照复印件', 'id': '49'}]}, {'name': '军队', 'id': '52', 'children': [{'name': '团级及以上证明', 'id': '53'}]}, {'name': '机关事业单位或其他非营利组织', 'id': '50', 'children': [{'name': '事业法人证书或组织机构代码证复印件', 'id': '51'}]}]}]}, {'name': '报装申请资料', 'id': '55', 'children': [{'name': '合法的土地使用证明', 'id': '60', 'children': [{'name': '军队', 'id': '64', 'children': [{'name': '团级及以上提供证明', 'id': '65'}]}, {'name': '农村', 'id': '62', 'children': [{'name': '由所在乡、镇或乡镇以上级别政府部门根据所辖权限开据证明', 'id': '63'}]}, {'name': '国有土地使用证，土地租赁协议及产权人同意报装证明材料', 'id': '61'}]}, {'name': '国家优惠电价', 'id': '66', 'children': [{'name': '政府有关部门核发的资质证明和公益证明（学校、养老院、居委会等）', 'id': '67'}]}, {'name': '客户用电报装申请书、客户用电设备信息登记表（动力、照明）', 'id': '56'}, {'name': '房屋产权证明', 'id': '57', 'children': [{'name': '租赁', 'id': '58', 'children': [{'name': '租赁协议复印件及产权人同意报装证明材料', 'id': '59'}]}]}]}, {'name': '项目立项及批复', 'id': '68', 'children': [{'name': '上级批准文件、立项批准文件', 'id': '69'}, {'name': '钢铁、电解铝、铁合金、水泥、电石、烧碱、黄磷、锌冶炼高耗能等特殊行业客户', 'id': '72', 'children': [{'name': '相关政府部门批准的许可文件包括政府主管部门立项或批复文件，环境评估报告、生产许可证等', 'id': '73'}]}, {'name': '需要规划立项', 'id': '70', 'children': [{'name': '建设工程规划许可证及附件', 'id': '71'}]}]}]}]}]}]}]}]
var ids = ["0"]

function deepCopy(target){
let copyed_objs = [];//此数组解决了循环引用和相同引用的问题，它存放已经递归到的目标对象
    function _deepCopy(target){
        if((typeof target !== 'object')||!target){return target;}
        for(let i= 0 ;i<copyed_objs.length;i++){
            if(copyed_objs[i].target === target){
                return copyed_objs[i].copyTarget;
            }
        }
        let obj = {};
        if(Array.isArray(target)){
            obj = [];//处理target是数组的情况
        }
        copyed_objs.push({target:target,copyTarget:obj})
        Object.keys(target).forEach(key=>{
            if(obj[key]){ return;}
            obj[key] = _deepCopy(target[key]);
        });
        return obj;
    }
    return _deepCopy(target);
}

function traverseTree(node){
        if (!node) {
            return;
        }
        if (ids.indexOf(node.id)>-1){
            console.log(node.name);
            node.label = {"backgroundColor":"green"}
        }

        if (node.children && node.children.length > 0) {
            var i = 0;
            for (i = 0; i < node.children.length; i++) {
                this.traverseTree(node.children[i]);
            }
        }

   }

function change_data(data){
    console.log("change data")
    traverseTree(data[0]); //注意js的语法 这里不用返回值 就能更新data2了
    console.log(data)
}


function y(s_data){
    // 指定图表的配置项和数据
    myChart.showLoading();

    myChart.hideLoading();
    myChart.setOption(option = {
    backgroundColor: '#02246d',
    tooltip: {
        trigger: 'item',
        formatter: '{b}'
    },
    legend: {
        top: '2%',
        left: '3%',
        bottom: '2%',
        orient: 'radial',
        data: [{
            name: 'KG',
            icon: 'rectangle'
        }],
        textStyle:{
            color:'#fff'
        }
    },
    series: [{
            type: 'tree',
            name: 'KG',
            data: s_data,
            top: '1%',
            right: '50%',
            orient: 'vertical',
            symbolSize: 1,
            initialTreeDepth: 10,
            label: {
                normal: {
                    position: 'center',
                    rotate: -90,
                    verticalAlign: 'middle',
                    align: 'left',
                    fontSize: 10,
                    backgroundColor: '#7049f0',
                    color: '#fff',
                    padding: 3,
                    formatter: [
                        '{box|{b}}'
                    ].join('\n'),
                    rich: {
                        box: {
                            height: 30,
                            color: '#fff',
                            padding: [0, 5],
                            align: 'center'
                        }
                    }
                }
            },

            expandAndCollapse: true,
            animationDuration: 550,
            animationDurationUpdate: 750
        }
    ]
});
}

y(data2)

    $(document).ready(function() {
        var S4 = function() {
           return (((1+Math.random())*0x10000)|0).toString(16).substring(1);
        };
        uid = (S4()+S4()+"-"+S4()+"-"+S4()+"-"+S4()+"-"+S4()+S4()+S4());
        var ws = new WebSocket("ws://" + window.location.host + '/ws/' + uid)

        show_log_flag = false;
        $("#show_log").click(function () {
            if (show_log_flag) {
                window.open("http://"+ window.location.host + "/log/" + uid + ".txt")
            } else {
                window.alert("多聊几句！")
            }
        })

        function add_item(cl, msg) {
            if (cl === 'right') {
                role = '用户'
            } else {
                role = '客服'
            }
            var template_html = '<li class="message '+ cl + ' appeared"><div class="avatar"><div class="text"><b>'+ role + '</b></div></div><div class="text_wrapper"><div class="text">' + msg + '</div></div></li>';
            $('.messages').append(template_html);
            $('.messages').animate({scrollTop: $('.messages').prop("scrollHeight")}, 500);
        }

        ws.onmessage = function(evt) {
            show_log_flag = true;
            // agent words 这里有个潜在问题是 如果msg本身有空格怎么办？可以在前期去除一下。。。TODO 如果传过来是空呢？应该没有
            //
            ids = evt.data.split(" ")
            var msg = ids.pop()

            add_item('left', msg);
            var new_data = deepCopy(data2)

//            color_id = [id.toString()]
            console.log("ids:"+ids)
            change_data(new_data)
            y(new_data)
        }

        function send_msg() {
            msg = $('.message_input').val()
            if (msg.length === 0) {
                console.log("不能发送空数据！")
                return;
            }
            add_item('right', msg)
            $(".message_input").val("")
            ws.send(msg);

        }
        $('.send_message').click(send_msg)
        $('.message_input').keypress(function(e) {
            if(e.which == 13) {
                send_msg()
            }
        })
    });

